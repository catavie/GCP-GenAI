LOCATION="us-central1"
MODEL_ID="gemini-1.0-pro-002"
PROJECT_ID="andrewcooley-test-project"
GENERATE_RESPONSE_METHOD="generateContent"

curl \
-X POST \
-H "Authorization: Bearer $(gcloud auth print-access-token)" \
-H "Content-Type: application/json" \
"https://${LOCATION}-aiplatform.googleapis.com/v1/projects/${PROJECT_ID}/locations/${LOCATION}/publishers/google/models:generateContent" -d \
{
  "contents": [
    {
      "role": "user",
      "parts": [
        {
          "text": "randomly select 10 words from a history book"
        }
      ]
    }
  ],
  "system_instruction":
    {
      "parts": [
        {
          "text": "please print the results in json format."
        }
      ]
    },
  "generation_config": {
    "maxOutputTokens": 2048,
    "temperature": 0.4,
    "topP": 1,
    "topK": 32
  }
}