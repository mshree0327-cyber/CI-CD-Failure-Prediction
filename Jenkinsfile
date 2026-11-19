
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

            curl -sS -f \
                http://host.docker.internal:8000/health

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
                echo "Check that Git history is not shallow."
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

            echo "Feature extraction completed."

            test -s build_features.json

            echo "Generated feature envelope:"
            cat build_features.json
        '''
    }

    stage('Validate 16 Model Features') {
        sh '''
            echo "Extracting and validating the 16 model features..."

            python3 - <<'PY'
import json

with open("build_features.json", "r", encoding="utf-8") as f:
    data = json.load(f)

features = data.get("model_features")

if not isinstance(features, dict):
    raise SystemExit(
        "ERROR: build_features.json does not contain a model_features object."
    )

if len(features) != 16:
    raise SystemExit(
        f"ERROR: Expected 16 features, received {len(features)}."
    )

with open("dynamic_prediction_input.json", "w", encoding="utf-8") as f:
    json.dump(features, f, indent=4)

print("Validated feature count:", len(features))
print("Feature names:")
for name in features:
    print("-", name)
PY

            echo "Dynamic prediction input:"
            cat dynamic_prediction_input.json
        '''
    }

    stage('Predict CI/CD Build') {
        sh '''
            echo "Sending 16 dynamic features to FastAPI..."

            curl -sS -f -X POST \
                http://host.docker.internal:8000/api/predict-dynamic \
                -H "Content-Type: application/json" \
                --data-binary @dynamic_prediction_input.json \
                -o prediction_result.json

            test -s prediction_result.json

            echo "Prediction result:"
            cat prediction_result.json

            echo ""
            echo "Dynamic prediction completed successfully."
        '''
    }

    stage('Archive Dynamic Prediction Results') {
        archiveArtifacts(
            artifacts: 'build_metadata.json,base_commit.txt,target_commit.txt,build_features.json,dynamic_prediction_input.json,prediction_result.json',
            fingerprint: true
        )

        echo 'All dynamic prediction artifacts archived successfully.'
    }
}
