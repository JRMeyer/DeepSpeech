#include "ctc_beam_search_decoder.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <map>
#include <utility>

#include "decoder_utils.h"
#include "ThreadPool.h"
#include "fst/fstlib.h"
#include "path_trie.h"

// helper function for kws_init
int setup_labels(const std::vector<int>& labels,
                 const int blank_id_,
                 std::vector<int>& labels_w_blanks,
                 std::vector<int>& s_inc,
                 std::vector<int>& e_inc) {
    const int L = labels.size();
    int repeats = 0;
    s_inc.push_back(1);
    for (int i = 1; i < L; ++i) {
        if (labels[i-1] == labels[i]) {
            s_inc.push_back(1);
            s_inc.push_back(1);
            e_inc.push_back(1);
            e_inc.push_back(1);
            ++repeats;
        }
        else {
            s_inc.push_back(2);
            e_inc.push_back(2);
        }
    }
    e_inc.push_back(1);

    for (int i = 0; i < L; ++i) {
        labels_w_blanks.push_back(blank_id_);
        labels_w_blanks.push_back(labels[i]);
    }
    labels_w_blanks.push_back(blank_id_);

    return repeats;
}


// helper function for kws_next
double log_add(double a, double b, double neginf) {
      if (a == neginf) return b;
      if (b == neginf) return a;
      if (a > b)
        return log1p(exp(b-a)) + a;
      else
        return log1p(exp(a-b)) + b;
}

// to be called from Decoder::init
int
DecoderState::kws_init(const Alphabet& alphabet,
                       const std::vector<int>& labels)
{
    blank_id_ = alphabet.GetSize();
    int repeats = setup_labels(labels,
                               blank_id_,
                               labels_w_blanks,
                               s_inc,
                               e_inc);
    S = labels_w_blanks.size();
    prev_alphas = new double[S];
    next_alphas = new double[S];
    neginf = -std::numeric_limits<double>::infinity();
    std::fill(prev_alphas, prev_alphas + S, neginf);
    return 0;
}

// to be called from Decoder::next
void
DecoderState::kws_next(const double* probs,
                       const int T,
                       const int alphabet_size)
{
    int kws_start =  (((S /2) + repeats - T) < 0) ? 0 : 1,
      kws_end = S > 1 ? 2 : 1;
    for (int i = kws_start; i < kws_end; ++i) {
        if (i == 0) {
            prev_alphas[i] = std::log(1 - probs[labels_w_blanks[1]]);
        } else {
            int l = labels_w_blanks[i];
            prev_alphas[i] = std::log(probs[l]);
        }
    }
    for(int t = 1; t < T; ++t) {
      std::fill(next_alphas, next_alphas + S, neginf);

      int remain = (S / 2) + repeats - (T - t);
      if(remain >= 0)
        kws_start += s_inc[remain];
      if(t <= (S / 2) + repeats)
        kws_end += e_inc[t - 1];
      int startloop = kws_start;
      int idx = t * alphabet_size;

      if (kws_start == 0) {
        double star_score = std::log(1 - probs[idx + labels_w_blanks[1]]);
        next_alphas[0] = prev_alphas[0] + star_score;
        startloop += 1;
      }

      for(int i = startloop; i < kws_end; ++i) {
        int l = labels_w_blanks[i];
        double prev_sum = log_add(prev_alphas[i], prev_alphas[i-1], neginf);

        // Skip two if not on blank and not on repeat.
        if (l != blank_id_ && i != 1 &&
            l != labels_w_blanks[i-2])
          prev_sum = log_add(prev_sum, prev_alphas[i-2], neginf);

        next_alphas[i] = prev_sum;
        if (i == labels_w_blanks.size() - 1) {
          double nl_score = probs[idx + labels_w_blanks[i-1]];
          next_alphas[i] += std::log(1 - nl_score);
        } else {
          next_alphas[i] += std::log(probs[l + idx]);
        }
      }
      std::swap(prev_alphas, next_alphas);
    }
}

int
DecoderState::init(const Alphabet& alphabet,
                   size_t beam_size,
                   double cutoff_prob,
                   size_t cutoff_top_n,
                   Scorer *ext_scorer)
{
  // assign special ids
  abs_time_step_ = 0;
  space_id_ = alphabet.GetSpaceLabel();
  blank_id_ = alphabet.GetSize();

  beam_size_ = beam_size;
  cutoff_prob_ = cutoff_prob;
  cutoff_top_n_ = cutoff_top_n;
  ext_scorer_ = ext_scorer;

  // init prefixes' root
  PathTrie *root = new PathTrie;
  root->score = root->log_prob_b_prev = 0.0;
  prefix_root_.reset(root);
  prefixes_.push_back(root);

  if (ext_scorer != nullptr) {
    // no need for std::make_shared<>() since Copy() does 'new' behind the doors
    auto dict_ptr = std::shared_ptr<PathTrie::FstType>(ext_scorer->dictionary->Copy(true));
    root->set_dictionary(dict_ptr);
    auto matcher = std::make_shared<fst::SortedMatcher<PathTrie::FstType>>(*dict_ptr, fst::MATCH_INPUT);
    root->set_matcher(matcher);
  }
  return 0;
}

void
DecoderState::next(const double *probs,
                   int time_dim,
                   int class_dim)
{
  // prefix search over time
  for (size_t rel_time_step = 0; rel_time_step < time_dim; ++rel_time_step, ++abs_time_step_) {
    auto *prob = &probs[rel_time_step*class_dim];

    float min_cutoff = -NUM_FLT_INF;
    bool full_beam = false;
    if (ext_scorer_ != nullptr) {
      size_t num_prefixes = std::min(prefixes_.size(), beam_size_);
      std::partial_sort(prefixes_.begin(),
                        prefixes_.begin() + num_prefixes,
                        prefixes_.end(),
                        prefix_compare);

      min_cutoff = prefixes_[num_prefixes - 1]->score +
                   std::log(prob[blank_id_]) - std::max(0.0, ext_scorer_->beta);
      full_beam = (num_prefixes == beam_size_);
    }

    std::vector<std::pair<size_t, float>> log_prob_idx =
        get_pruned_log_probs(prob, class_dim, cutoff_prob_, cutoff_top_n_);
    // loop over class dim
    for (size_t index = 0; index < log_prob_idx.size(); index++) {
      auto c = log_prob_idx[index].first;
      auto log_prob_c = log_prob_idx[index].second;

      for (size_t i = 0; i < prefixes_.size() && i < beam_size_; ++i) {
        auto prefix = prefixes_[i];
        if (full_beam && log_prob_c + prefix->score < min_cutoff) {
          break;
        }

        // blank
        if (c == blank_id_) {
          prefix->log_prob_b_cur =
              log_sum_exp(prefix->log_prob_b_cur, log_prob_c + prefix->score);
          continue;
        }

        // repeated character
        if (c == prefix->character) {
          prefix->log_prob_nb_cur = log_sum_exp(
              prefix->log_prob_nb_cur, log_prob_c + prefix->log_prob_nb_prev);
        }

        // get new prefix
        auto prefix_new = prefix->get_path_trie(c, abs_time_step_, log_prob_c);

        if (prefix_new != nullptr) {
          float log_p = -NUM_FLT_INF;

          if (c == prefix->character &&
              prefix->log_prob_b_prev > -NUM_FLT_INF) {
            log_p = log_prob_c + prefix->log_prob_b_prev;
          } else if (c != prefix->character) {
            log_p = log_prob_c + prefix->score;
          }

          if (ext_scorer_ != nullptr) {
            // skip scoring the space in word based LMs
            PathTrie* prefix_to_score;
            if (ext_scorer_->is_utf8_mode()) {
              prefix_to_score = prefix_new;
            } else {
              prefix_to_score = prefix;
            }

            // language model scoring
            if (ext_scorer_->is_scoring_boundary(prefix_to_score, c)) {
              float score = 0.0;
              std::vector<std::string> ngram;
              ngram = ext_scorer_->make_ngram(prefix_to_score);
              bool bos = ngram.size() < ext_scorer_->get_max_order();
              score = ext_scorer_->get_log_cond_prob(ngram, bos) * ext_scorer_->alpha;
              log_p += score;
              log_p += ext_scorer_->beta;
            }
          }

          prefix_new->log_prob_nb_cur =
              log_sum_exp(prefix_new->log_prob_nb_cur, log_p);
        }
      }  // end of loop over prefix
    }    // end of loop over alphabet

    // update log probs
    prefixes_.clear();
    prefix_root_->iterate_to_vec(prefixes_);

    // only preserve top beam_size prefixes
    if (prefixes_.size() > beam_size_) {
      std::nth_element(prefixes_.begin(),
                       prefixes_.begin() + beam_size_,
                       prefixes_.end(),
                       prefix_compare);
      for (size_t i = beam_size_; i < prefixes_.size(); ++i) {
        prefixes_[i]->remove();
      }

      // Remove the elements from std::vector
      prefixes_.resize(beam_size_);
    }
  }  // end of loop over time
}

double
DecoderState::kws_decode() const
{
    double loglike = neginf;
    for(int i = kws_start; i < kws_end; ++i) {
      loglike = log_add(loglike, prev_alphas[i], neginf);
    }
    // Cleanup
    delete[] prev_alphas;
    delete[] next_alphas;

    return -loglike;
}

std::vector<Output>
DecoderState::decode() const
{
  std::vector<PathTrie*> prefixes_copy = prefixes_;
  std::unordered_map<const PathTrie*, float> scores;
  for (PathTrie* prefix : prefixes_copy) {
    scores[prefix] = prefix->score;
  }

  // score the last word of each prefix that doesn't end with space
  if (ext_scorer_ != nullptr) {
    for (size_t i = 0; i < beam_size_ && i < prefixes_copy.size(); ++i) {
      auto prefix = prefixes_copy[i];
      if (!ext_scorer_->is_scoring_boundary(prefix->parent, prefix->character)) {
        float score = 0.0;
        std::vector<std::string> ngram = ext_scorer_->make_ngram(prefix);
        bool bos = ngram.size() < ext_scorer_->get_max_order();
        score = ext_scorer_->get_log_cond_prob(ngram, bos) * ext_scorer_->alpha;
        score += ext_scorer_->beta;
        scores[prefix] += score;
      }
    }
  }

  using namespace std::placeholders;
  size_t num_prefixes = std::min(prefixes_copy.size(), beam_size_);
  std::partial_sort(prefixes_copy.begin(),
                    prefixes_copy.begin() + num_prefixes,
                    prefixes_copy.end(),
                    std::bind(prefix_compare_external, _1, _2, scores));

  //TODO: expose this as an API parameter
  const size_t top_paths = 1;
  size_t num_returned = std::min(num_prefixes, top_paths);

  std::vector<Output> outputs;
  outputs.reserve(num_returned);

  // compute aproximate ctc score as the return score, without affecting the
  // return order of decoding result. To delete when decoder gets stable.
  for (size_t i = 0; i < num_returned; ++i) {
    Output output;
    prefixes_copy[i]->get_path_vec(output.tokens, output.timesteps);
    double approx_ctc = scores[prefixes_copy[i]];
    if (ext_scorer_ != nullptr) {
      auto words = ext_scorer_->split_labels_into_scored_units(output.tokens);
      // remove term insertion weight
      approx_ctc -= words.size() * ext_scorer_->beta;
      // remove language model weight
      approx_ctc -= (ext_scorer_->get_sent_log_prob(words)) * ext_scorer_->alpha;
    }
    output.confidence = -approx_ctc;
    outputs.push_back(output);
  }

  return outputs;
}

std::vector<Output> ctc_beam_search_decoder(
    const double *probs,
    int time_dim,
    int class_dim,
    const Alphabet &alphabet,
    size_t beam_size,
    double cutoff_prob,
    size_t cutoff_top_n,
    Scorer *ext_scorer)
{
  DecoderState state;
  state.init(alphabet, beam_size, cutoff_prob, cutoff_top_n, ext_scorer);
  state.next(probs, time_dim, class_dim);
  return state.decode();
}


double kws_decoder(
    const double *probs,
    int time_dim,
    int class_dim,
    const Alphabet &alphabet,
    size_t beam_size,
    double cutoff_prob,
    size_t cutoff_top_n,
    Scorer *ext_scorer)
{
  DecoderState state;
  // init kws variables
  const std::vector<int>& keyword = {1,2,3};
  state.kws_init(alphabet, keyword);
  // next kws
  int T = 1000;
  const int alphabet_size = 29;
  state.kws_next(probs,T,alphabet_size);
  return state.kws_decode();
}





std::vector<std::vector<Output>>
ctc_beam_search_decoder_batch(
    const double *probs,
    int batch_size,
    int time_dim,
    int class_dim,
    const int* seq_lengths,
    int seq_lengths_size,
    const Alphabet &alphabet,
    size_t beam_size,
    size_t num_processes,
    double cutoff_prob,
    size_t cutoff_top_n,
    Scorer *ext_scorer)
{
  VALID_CHECK_GT(num_processes, 0, "num_processes must be nonnegative!");
  VALID_CHECK_EQ(batch_size, seq_lengths_size, "must have one sequence length per batch element");
  // thread pool
  ThreadPool pool(num_processes);

  // enqueue the tasks of decoding
  std::vector<std::future<std::vector<Output>>> res;
  for (size_t i = 0; i < batch_size; ++i) {
    res.emplace_back(pool.enqueue(ctc_beam_search_decoder,
                                  &probs[i*time_dim*class_dim],
                                  seq_lengths[i],
                                  class_dim,
                                  alphabet,
                                  beam_size,
                                  cutoff_prob,
                                  cutoff_top_n,
                                  ext_scorer));
  }

  // get decoding results
  std::vector<std::vector<Output>> batch_results;
  for (size_t i = 0; i < batch_size; ++i) {
    batch_results.emplace_back(res[i].get());
  }
  return batch_results;
}
