# Corporate Certificates Directory

## Purpose

This directory is for corporate SSL certificates needed for Docker containers to access external resources (like HuggingFace) in corporate networks.

## How to Use

1. **Get your certificate** from IT department (should be a `.crt` file)

2. **Place the certificate here:**
   ```
   certs/
   ├── README.md          ← This file
   └── corporate.crt      ← Your certificate (any name ending in .crt)
   ```

3. **Rebuild Docker images:**
   ```bash
   docker-compose build
   ```

## Multiple Certificates

You can place multiple `.crt` files here if needed. All `.crt` files in this directory will be installed in the Docker containers.

## Security

- Certificate files (`*.crt`) are automatically ignored by git (.gitignore)
- Certificates are only copied into Docker images during build
- They are never committed to the repository

## No Certificate Needed?

If you don't have corporate firewall restrictions, you can leave this directory empty (except for this README). The Docker build will work fine.
