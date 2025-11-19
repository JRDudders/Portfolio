# Corporate Certificate Setup for Docker

If your corporate network requires a custom SSL certificate, follow these steps:

## Step 1: Get Your Corporate Certificate

Ask your IT department for the corporate root CA certificate. It should be a `.crt` file.

## Step 2: Place the Certificate

Rename your certificate file to `corporate-cert.crt` and place it in the project root:

```
Sentiment Docker Test/
├── corporate-cert.crt    ← Place your certificate here
├── Dockerfile.nlp
├── docker-compose.yml
└── ...
```

## Step 3: Rebuild Docker Images

The Dockerfiles are already configured to automatically install the certificate if present.

```bash
# Rebuild the NLP service
docker-compose build nlp

# Or rebuild all services
docker-compose build
```

## Step 4: Start the Services

```bash
docker-compose up
```

The certificate will be installed in the container's trust store and Python will use it automatically.

## Verification

Check that the certificate is installed:

```bash
# Enter the container
docker-compose exec nlp bash

# List installed certificates
ls -la /usr/local/share/ca-certificates/

# Check Python can use it
python -c "import requests; print(requests.get('https://huggingface.co').status_code)"
```

You should see `200` if the certificate is working correctly.

## Troubleshooting

**Certificate not found during build:**
- Make sure the file is named exactly `corporate-cert.crt`
- Make sure it's in the project root directory
- Check `.dockerignore` doesn't exclude it

**Still getting SSL errors:**
- Verify the .crt file is valid (open it in a text editor, should start with `-----BEGIN CERTIFICATE-----`)
- Ask IT if you need multiple certificates (chain of trust)
- Check if you need to set additional environment variables

**Multiple certificates:**
If you need to install multiple certificates, you can:
1. Rename them to `corporate-cert-1.crt`, `corporate-cert-2.crt`, etc.
2. Update the Dockerfile COPY line to include all of them:
```dockerfile
COPY corporate-cert*.crt /usr/local/share/ca-certificates/
```

## Security Note

**DO NOT commit the certificate file to git!** It's already in `.gitignore`.

The certificate is only copied into the Docker image during build and is not exposed outside the container.
