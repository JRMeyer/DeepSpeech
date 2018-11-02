import pandas
import sys
import os

LOCALE = 'ky'
output_folder = sys.argv[1]

clips = pandas.read_csv('/snakepit/shared/data/mozilla/CommonVoice/v2.0-alpha1.0/clips.tsv', sep='\t')
clips['path'] = clips['path'].str.replace('/', '___')

locale = clips[clips['locale'] == LOCALE]
locale_dev_paths = locale[locale['bucket'] == 'dev'].loc[:, ['path']]
locale_test_paths = locale[locale['bucket'] == 'test'].loc[:, ['path']]

mixed = pandas.read_csv('/snakepit/shared/data/mozilla/CommonVoice/v2.0-alpha1.0/{}/cv_ky_valid.csv'.format(LOCALE))

locale_dev_paths['path'] = '/snakepit/shared/data/mozilla/CommonVoice/v2.0-alpha1.0/{}/valid/'.format(LOCALE) + locale_dev_paths['path'].astype(str)
locale_test_paths['path'] = '/snakepit/shared/data/mozilla/CommonVoice/v2.0-alpha1.0/{}/valid/'.format(LOCALE) + locale_test_paths['path'].astype(str)

dev_indices = mixed['wav_filename'].isin(locale_dev_paths['path'])
test_indices = mixed['wav_filename'].isin(locale_test_paths['path'])
train_indices = ~(dev_indices | test_indices)

mixed[dev_indices].to_csv(os.path.join(output_folder, 'cv_{}_valid_dev.csv'.format(LOCALE)), index=False)
mixed[test_indices].to_csv(os.path.join(output_folder, 'cv_{}_valid_test.csv'.format(LOCALE)), index=False)
mixed[train_indices].to_csv(os.path.join(output_folder, 'cv_{}_valid_train.csv'.format(LOCALE)), index=False)
