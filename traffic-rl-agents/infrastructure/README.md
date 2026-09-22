# Infrastructure

AWS infrastructure for TrafficRL, managed via CloudFormation.

## Architecture

```
Internet
  │
  ├── GitHub Actions (OIDC)
  │     ├── ECR: traffic-rl image
  │     └── S3: traffic-rl-artifacts (training output, model checkpoints)
  │
  └── SageMaker
        ├── Training Jobs (on-demand or spot)
        │     ├── ECR image
        │     ├── S3 input (scenarios, configs)
        │     └── S3 output (model artifacts)
        │
        ├── Hyperparameter Tuning (Bayesian HPO)
        │
        └── Endpoints (real-time inference)
              ├── ECR model image
              └── S3 model artifacts
```

## Files

| File                     | Description                          |
|--------------------------|--------------------------------------|
| `cloudformation.yaml`    | Core stack: ECR, S3, IAM             |
| `cloudformation-sagemaker.yaml` | SageMaker resources           |
| `iam-policies/`          | IAM policy documents                 |

## Deployment

```bash
# Deploy core infrastructure
aws cloudformation deploy \
  --template-file infrastructure/cloudformation.yaml \
  --stack-name traffic-rl-core \
  --capabilities CAPABILITY_IAM

# Deploy SageMaker resources
aws cloudformation deploy \
  --template-file infrastructure/cloudformation-sagemaker.yaml \
  --stack-name traffic-rl-sagemaker \
  --capabilities CAPABILITY_IAM
```

## CI/CD

GitHub Actions uses OIDC to authenticate to AWS without static keys:

- `AWS_ROLE_ARN` env var → GitHub OIDC role
- Secrets from AWS Secrets Manager at runtime
- Container built and pushed to ECR on tag pushes
- Training deployed to SageMaker via the release workflow
