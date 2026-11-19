
import groovy.json.JsonOutput

node {
    stage('Checkout GitHub Repository') {
        checkout scm
        echo 'GitHub repository checked out successfully!'
    }

    stage('Record Build Metadata') {
        def metadata = [
            jenkins_build_number: env.BUILD_NUMBER,
            jenkins_build_id: env.BUILD_ID,
            git_commit: sh(
                script: 'git rev-parse HEAD',
                returnStdout: true
            ).trim(),
            git_branch: env.GIT_BRANCH ?: 'unknown',
            git_url: sh(
                script: 'git config --get remote.origin.url',
                returnStdout: true
            ).trim(),
            commit_author: sh(
                script: 'git log -1 --format=%an',
                returnStdout: true
            ).trim(),
            commit_author_email: sh(
                script: 'git log -1 --format=%ae',
                returnStdout: true
            ).trim(),
            commit_timestamp: sh(
                script: 'git log -1 --format=%aI',
                returnStdout: true
            ).trim(),
            commit_message: sh(
                script: 'git log -1 --format=%s',
                returnStdout: true
            ).trim(),
            metadata_recorded_at_utc: new Date().format(
                "yyyy-MM-dd'T'HH:mm:ss'Z'",
                TimeZone.getTimeZone('UTC')
            )
        ]

        writeFile(
            file: 'build_metadata.json',
            text: JsonOutput.prettyPrint(JsonOutput.toJson(metadata))
        )

        archiveArtifacts(
            artifacts: 'build_metadata.json',
            fingerprint: true
        )

        echo "Metadata recorded for commit: ${metadata.git_commit}"
    }

    stage('Check FastAPI') {
        sh '''
            echo "Checking FastAPI health..."
            curl -sS -f http://host.docker.internal:8000/health
            echo ""
            echo "FastAPI is available."
        '''
    }

    stage('Identify Git Commits') {
        sh '''
            echo "Identifying target and base commits..."

            TARGET=$(git rev-parse HEAD)

            if ! BASE=$(git rev-parse "${TARGET}^" 2>/dev/null); then
                echo "ERROR: Parent commit not available."
                exit 1
            fi

            git cat-file -e "${BASE}^{commit}"
            git cat-file -e "${TARGET}^{commit}"

            echo "$BASE" > base_commit.txt
            echo "$TARGET" > target_commit.txt

            echo "BASE: $BASE"
            echo "TARGET: $TARGET"
        '''
    }

    stage('Generate Dynamic Features') {
        sh '''
            echo "Generating dynamic pre-build features..."

            BASE=$(cat base_commit.txt)
            TARGET=$(cat target_commit.txt)

            python3 backend/live_feature_generator.py \
                --base "$BASE" \
                --target "$TARGET" \
                --output build_features.json

            test -s build_features.json

            echo "Feature extraction completed."
        '''
    }

    stage('Validate 16 Model Features') {
        sh '''
            python3 - <<'PY'
import json

with open("build_features.json", encoding="utf-8") as f:
    data = json.load(f)

features = data.get("model_features")

if not isinstance(features, dict):
    raise SystemExit("ERROR: model_features object missing.")

if len(features) != 16:
    raise SystemExit(
        f"ERROR: Expected 16 features, received {len(features)}."
    )

with open("dynamic_prediction_input.json", "w", encoding="utf-8") as f:
    json.dump(features, f, indent=4)

print("Validated feature count:", len(features))
PY
        '''
    }

    stage('Predict CI/CD Build') {
        sh '''
            echo "Sending dynamic features to FastAPI..."

            curl -sS -f -X POST \
                http://host.docker.internal:8000/api/predict-dynamic \
                -H "Content-Type: application/json" \
                --data-binary @dynamic_prediction_input.json \
                -o prediction_result.json

            test -s prediction_result.json

            echo "Prediction result:"
            cat prediction_result.json
        '''
    }

    stage('Build Python Application') {
        script {
            int buildStatus = sh(
                script: '''
                    echo "Compiling the Python application..."

                    python3 -m compileall -q demo_app
                ''',
                returnStatus: true
            )

            writeFile(
                file: 'application_build_exit_code.txt',
                text: "${buildStatus}"
            )

            echo "Application build exit code: ${buildStatus}"
        }
    }

    stage('Run Pytest') {
        script {
            int testStatus = sh(
                script: '''
                    echo "Running application tests..."

                    /var/jenkins_home/pytest-venv/bin/python \
                        -m pytest demo_app -v
                ''',
                returnStatus: true
            )

            writeFile(
                file: 'pytest_exit_code.txt',
                text: "${testStatus}"
            )

            echo "Pytest exit code: ${testStatus}"
        }
    }

    stage('Capture Actual Build Outcome') {
        script {
            int buildStatus = readFile(
                'application_build_exit_code.txt'
            ).trim().toInteger()

            int testStatus = readFile(
                'pytest_exit_code.txt'
            ).trim().toInteger()

            def actualOutcome =
                (buildStatus == 0 && testStatus == 0) ?
                'SUCCESS' : 'FAILURE'

            def result = [
                application_build_status: buildStatus == 0 ? 'SUCCESS' : 'FAILURE',
                pytest_status: testStatus == 0 ? 'SUCCESS' : 'FAILURE',
                application_build_exit_code: buildStatus,
                pytest_exit_code: testStatus,
                actual_outcome: actualOutcome,
                recorded_at_utc: new Date().format(
                    "yyyy-MM-dd'T'HH:mm:ss'Z'",
                    TimeZone.getTimeZone('UTC')
                )
            ]

            writeFile(
                file: 'actual_build_outcome.json',
                text: JsonOutput.prettyPrint(JsonOutput.toJson(result))
            )

            echo "Actual application outcome: ${actualOutcome}"

            if (actualOutcome == 'FAILURE') {
                currentBuild.result = 'UNSTABLE'
            }
        }
    }

    stage('Compare Prediction vs Actual') {
        sh '''
            python3 - <<'PY'
import json

with open("prediction_result.json", encoding="utf-8") as f:
    prediction_data = json.load(f)

with open("actual_build_outcome.json", encoding="utf-8") as f:
    actual_data = json.load(f)

predicted = str(
    prediction_data.get("prediction", "UNKNOWN")
).upper()

actual = str(
    actual_data.get("actual_outcome", "UNKNOWN")
).upper()

comparison = {
    "predicted_outcome": predicted,
    "actual_outcome": actual,
    "prediction_correct": None,
    "confusion_matrix_category": "NOT_AVAILABLE"
}

if predicted in ("SUCCESS", "FAILURE") and actual in ("SUCCESS", "FAILURE"):
    comparison["prediction_correct"] = predicted == actual

    if predicted == "FAILURE" and actual == "FAILURE":
        category = "TRUE_POSITIVE"
    elif predicted == "SUCCESS" and actual == "SUCCESS":
        category = "TRUE_NEGATIVE"
    elif predicted == "FAILURE" and actual == "SUCCESS":
        category = "FALSE_POSITIVE"
    else:
        category = "FALSE_NEGATIVE"

    comparison["confusion_matrix_category"] = category

with open("prediction_actual_comparison.json", "w", encoding="utf-8") as f:
    json.dump(comparison, f, indent=4)

print(json.dumps(comparison, indent=4))
PY
        '''
    }

    stage('Archive All Results') {
        archiveArtifacts(
            artifacts: 'build_metadata.json,base_commit.txt,target_commit.txt,build_features.json,dynamic_prediction_input.json,prediction_result.json,application_build_exit_code.txt,pytest_exit_code.txt,actual_build_outcome.json,prediction_actual_comparison.json',
            fingerprint: true
        )

        echo 'All prediction, build, test and comparison artifacts archived.'
    }
}
