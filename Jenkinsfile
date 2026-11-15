
import groovy.json.JsonOutput

node {
    stage('Checkout GitHub Repository') {
        checkout scm
        echo 'GitHub repository checked out successfully!'
    }

    // NEW: Record actual Jenkins and Git metadata
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
            text: JsonOutput.prettyPrint(
                JsonOutput.toJson(metadata)
            )
        )

        archiveArtifacts(
            artifacts: 'build_metadata.json',
            fingerprint: true
        )

        echo 'Build metadata recorded successfully!'
        echo "Git commit: ${metadata.git_commit}"
    }

    stage('Test Jenkins') {
        echo 'Jenkins is working successfully!'
    }

    stage('Check FastAPI') {
        sh '''
            echo "Checking FastAPI..."

            curl -sS -f \
            http://host.docker.internal:8000/health

            echo ""
        '''
    }

    stage('Generate Build Features') {
        sh '''
            echo "Generating 19 pre-build features..."

            cat > build_features.json <<'EOF'
{
    "git_diff_src_churn": 0,
    "gh_diff_files_added": 0,
    "gh_diff_files_deleted": 0,
    "gh_diff_files_modified": 2,
    "gh_diff_src_files": 0,
    "gh_diff_doc_files": 0,
    "gh_diff_other_files": 0,
    "gh_sloc": 1110,
    "gh_num_commits_on_files_touched": 28,
    "code_churn_rate": 0,
    "files_changed_total": 2,
    "git_prev_commit_resolution_status": "build_found",
    "gh_diff_tests_added": 0,
    "gh_diff_tests_deleted": 0,
    "gh_test_lines_per_kloc": 2854.054054,
    "gh_test_cases_per_kloc": 217.117117,
    "gh_asserts_cases_per_kloc": 275.675676,
    "gh_team_size": 5,
    "gh_by_core_team_member": 0
}
EOF

            echo "Validating JSON feature count..."

            FEATURE_COUNT=$(grep -c ':' build_features.json)
            echo "Feature count: $FEATURE_COUNT"
            test "$FEATURE_COUNT" -eq 19

            echo "Generated build features:"
            cat build_features.json
        '''
    }

    stage('Predict CI/CD Build') {
        sh '''
            echo "Sending 19 features to ML API..."

            curl -sS -f -X POST \
            http://host.docker.internal:8000/api/predict \
            -H "Content-Type: application/json" \
            --data-binary @build_features.json

            echo ""

            echo "Prediction completed successfully."
        '''
    }
}
