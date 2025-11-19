# Corporate Certificate Setup for Docker

If your corporate network requires a custom SSL certificate, follow these steps:

## Step 1: Get Your Corporate Certificate

Ask your IT department for the corporate root CA certificate. It should be a `.crt` file.

## Step 2: Place the Certificate

Place your certificate file(s) in the `certs/` directory:

```
Sentiment Docker Test/
├── certs/
│   ├── README.md         ← Instructions (keep this)
│   └── corporate.crt     ← Your certificate (any name ending in .crt)
├── Dockerfile.nlp
├── docker-compose.yml
└── ...
```

**Note:** You can use any filename ending in `.crt`. Multiple certificates are supported.

## Step 3: Rebuild Docker Images

The Dockerfiles are already configured to automatically install all certificates from the `certs/` directory.

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
- Make sure the .crt file is in the `certs/` directory
- Make sure the filename ends with `.crt`
- The directory must exist (it should already be in the repo)

**Still getting SSL errors:**
- Verify the .crt file is valid (open it in a text editor, should start with `-----BEGIN CERTIFICATE-----`)
- Ask IT if you need multiple certificates (chain of trust)
- Try rebuilding without cache: `docker-compose build --no-cache nlp`

**Multiple certificates:**
If you have a certificate chain, place all `.crt` files in the `certs/` directory:
```
certs/
├── README.md
├── root-ca.crt
├── intermediate-ca.crt
└── corporate.crt
```

All `.crt` files will be automatically installed.

## Security Note

**DO NOT commit certificate files to git!** They are already ignored by `.gitignore`.

The certificates are only copied into the Docker image during build and are not exposed outside the container.

## No Certificate Needed?

If you don't have corporate firewall restrictions, you can leave the `certs/` directory empty (except for README.md). The Docker build will work fine and simply skip the certificate installation step.
