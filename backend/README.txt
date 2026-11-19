COMMIT-SNAPSHOT STATIC EXTRACTOR UPDATE

Replace these two files in your backend folder:
- static_feature_extractor.py
- live_feature_generator.py

Keep your current git_feature_extractor.py unchanged for this test.
Your existing backend_backup_dynamic folder remains your rollback copy.

The static extractor now reads tracked Python files from the requested TARGET
commit using Git, rather than scanning the current working directory. It no
longer excludes the entire extractor_test folder. It still excludes generated
and cache directories (venv, .git, __pycache__, etc.).

Run from the repository root:
python backend\live_feature_generator.py --base c99ee80 --target e5d36b2

Then display the feature values:
python -c "import json; f=json.load(open('build_features.json')); print('FEATURES'); [print(k,':',v) for k,v in f['model_features'].items()]; print('STATIC DIAGNOSTICS:', f['static_diagnostics']['diagnostics'])"

Expected for this controlled commit:
- Python source/test files are read from commit e5d36b2.
- extractor_test/tests/test_sample.py is no longer excluded merely because it
  is under extractor_test.
- Test metrics may be nonzero if that commit contains test_* functions and
  assert statements.
- Static metrics are now tied to e5d36b2, not the current HEAD.

Research limitation:
These remain Python-only operational definitions. Do not claim exact historical
TravisTorrent feature parity. Validate against training distributions before
connecting the model to FastAPI or Jenkins.
