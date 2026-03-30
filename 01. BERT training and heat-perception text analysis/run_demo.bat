@echo off
REM run_demo.bat
REM End-to-end pipeline demonstration using the synthetic sample dataset.
REM Requires: pip install -r requirements.txt
REM
REM Usage:
REM   run_demo.bat
REM
REM All outputs written to demo_outputs\

setlocal enabledelayedexpansion

set DEMO_CSV=sample_data\demo_tweets.csv
set OUT=demo_outputs

echo ============================================================
echo  Heat Perception Pipeline -- End-to-End Demo
echo ============================================================
echo  Input : %DEMO_CSV%  (620 synthetic labelled examples)
echo  Output: %OUT%\
echo.

REM Step 0 - re-generate demo data
echo [00] Verifying demo dataset reproducibility...
python sample_data\generate_demo_sample.py --output_dir sample_data --seed 42
if errorlevel 1 goto :error
echo      demo_tweets.csv verified.
echo.

REM Step 1 - Candidate retrieval
echo [01] Candidate retrieval (keyword + negation)...
python 01_candidate_retrieval.py --input_csv %DEMO_CSV% --output_csv %OUT%\01_candidates.csv --keep_all_rows
if errorlevel 1 goto :error
echo.

REM Step 2 - Rule filtering
echo [02] Rule-based exclusion filtering...
python 02_rule_filtering.py --input_csv %OUT%\01_candidates.csv --output_csv %OUT%\02_filtered.csv --audit_csv %OUT%\02_audit.csv --keep_all_rows
if errorlevel 1 goto :error
echo.

REM Step 3 - Deduplication
echo [03] Deduplication and account hygiene...
python 03_deduplicate_and_account_hygiene.py --input_csv %OUT%\02_filtered.csv --output_csv %OUT%\03_deduped.csv --removal_log_csv %OUT%\03_removal_log.csv
if errorlevel 1 goto :error
echo.

REM Step 4 - Gold-label schema
echo [04] Building gold-label schema...
python 04_build_gold_labels.py --input_csv %OUT%\03_deduped.csv --output_csv %OUT%\04_gold_labeled.csv --augment_boundary_examples
if errorlevel 1 goto :error
echo.

REM Step 5 - BERT training
echo [05] BERT training with grouped 3-fold CV...
python 05_train_grouped_cv.py --data_path %OUT%\04_gold_labeled.csv --output_dir %OUT%\05_model --model_name_or_path bert-base-uncased --num_folds 3 --num_train_epochs 2 --temporal_test_csv %DEMO_CSV%
if errorlevel 1 goto :error
echo.

REM Step 6 - Validation
echo [06] Validation and monthly audit...
python 06_validate_and_audit.py --model_dir %OUT%\05_model\final_model --input_csv %OUT%\04_gold_labeled.csv --output_dir %OUT%\06_audit --audit_per_month 50
if errorlevel 1 goto :error
echo.

REM Step 7 - Aggregate indices
echo [07] Aggregating city-day HPII / HPVI / HPPI indices...
python 07_aggregate_city_day_indices.py --input_csv %OUT%\06_audit\predictions.csv --output_dir %OUT%\07_indices
if errorlevel 1 goto :error
echo.

echo ============================================================
echo  Demo complete.  All outputs in: %OUT%\
echo.
echo  Key files:
echo    %OUT%\05_model\monthly_test_metrics.csv
echo    %OUT%\06_audit\overall_metrics.json
echo    %OUT%\07_indices\
echo.
echo  Reference performance (from real corpus):
echo    sample_outputs\monthly_test_metrics_reference.csv
echo ============================================================
goto :end

:error
echo.
echo ERROR: a pipeline step failed. See output above.
exit /b 1

:end
endlocal
