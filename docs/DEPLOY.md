# Deploy Gemma 4 to SageMaker with the aws CLI

Nine steps, all plain `aws` commands. Example: `google/gemma-4-E2B-it` on one
NVIDIA L4 in us-east-1. The same steps are wrapped by `make deploy` / `sm.py`.

```bash
export AWS_REGION=us-east-1 AWS_PAGER=
export NAME=gemma-4-e2b
export MODEL_ID=google/gemma-4-E2B-it
```

#### Step 1 — Log in

```bash
aws login
aws sts get-caller-identity
```

#### Step 2 — Check the endpoint quota for your instance type

```bash
aws service-quotas list-service-quotas --service-code sagemaker \
  --query "Quotas[?QuotaName=='ml.g6.xlarge for endpoint usage'].Value"
```

`0` means request an increase in the Service Quotas console first.

#### Step 3 — Find the vLLM container image

AWS publishes a SageMaker build of vLLM. Take the highest version among the
`*-sagemaker-vN` tags (older lines get patch rebuilds, so sort by version).

```bash
TAG=$(aws ecr describe-images --registry-id 763104351884 --repository-name vllm \
  --query "imageDetails[].imageTags[]" --output text | tr '\t' '\n' \
  | grep -E -- '^[0-9]+\.[0-9]+\.[0-9]+-.*-sagemaker-v[0-9]+\.[0-9]+$' | sort -V | tail -1)
export IMAGE=763104351884.dkr.ecr.$AWS_REGION.amazonaws.com/vllm:$TAG
echo $IMAGE   # vllm:0.30.0-gpu-py312-cu130-ubuntu24.04-sagemaker-v1.1 on 2026-09-25
```

#### Step 4 — Create the execution role (once per account)

```bash
aws iam create-role --role-name sagemaker-gemma-execution-role \
  --assume-role-policy-document '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"sagemaker.amazonaws.com"},"Action":"sts:AssumeRole"}]}'
aws iam attach-role-policy --role-name sagemaker-gemma-execution-role \
  --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess
export ROLE=$(aws iam get-role --role-name sagemaker-gemma-execution-role --query Role.Arn --output text)
```

#### Step 5 — Create the model

The container reads vLLM flags from `SM_VLLM_*` environment variables and
downloads the weights from Hugging Face at start-up. Gemma 4 is Apache-2.0 and
ungated, so no token is needed.

```bash
aws sagemaker create-model --model-name $NAME --execution-role-arn $ROLE \
  --primary-container "{\"Image\":\"$IMAGE\",\"Environment\":{
    \"SM_VLLM_MODEL\":\"$MODEL_ID\",
    \"SM_VLLM_MAX_MODEL_LEN\":\"8192\",
    \"SM_VLLM_GPU_MEMORY_UTILIZATION\":\"0.9\"}}"
```

#### Step 6 — Create the endpoint config

`InstancePools` lists fallback instance types in priority order, so SageMaker
tries the next one when a type has no capacity. All three below carry one L4.

```bash
aws sagemaker create-endpoint-config --endpoint-config-name $NAME \
  --production-variants "[{\"VariantName\":\"AllTraffic\",\"ModelName\":\"$NAME\",
    \"InitialInstanceCount\":1,
    \"InstancePools\":[{\"InstanceType\":\"ml.g6.xlarge\",\"Priority\":1},
                       {\"InstanceType\":\"ml.g6.2xlarge\",\"Priority\":2},
                       {\"InstanceType\":\"ml.g6.4xlarge\",\"Priority\":3}],
    \"ContainerStartupHealthCheckTimeoutInSeconds\":1800,
    \"ModelDataDownloadTimeoutInSeconds\":1800}]"
```

For a single type, replace `InstancePools` with `"InstanceType":"ml.g6.xlarge"`.

#### Step 7 — Create the endpoint and wait

```bash
aws sagemaker create-endpoint --endpoint-name $NAME --endpoint-config-name $NAME
aws sagemaker wait endpoint-in-service --endpoint-name $NAME
aws sagemaker describe-endpoint --endpoint-name $NAME --query '[EndpointStatus,FailureReason]'
```

Billing starts once an instance is placed. Follow start-up in
`/aws/sagemaker/Endpoints/$NAME` (`aws logs tail /aws/sagemaker/Endpoints/$NAME --follow`).

#### Step 8 — Call it

```bash
echo '{"messages":[{"role":"user","content":"Why is the sky blue?"}],"max_tokens":256}' > req.json
aws sagemaker-runtime invoke-endpoint --endpoint-name $NAME \
  --content-type application/json --body fileb://req.json out.json
jq -r '.choices[0].message.content' out.json
```

#### Step 9 — Delete it (stops billing)

```bash
aws sagemaker delete-endpoint --endpoint-name $NAME
aws sagemaker delete-endpoint-config --endpoint-config-name $NAME
aws sagemaker delete-model --model-name $NAME
```

---

#### 🔎 Tip: capacity

A quota of 1 lets you request one instance; the region still has to have one
free. SageMaker reports `InsufficientInstanceCapacity` about 30 minutes after
the request, and no log group appears while it waits. Use `InstancePools`
(Step 6), or repeat Steps 5–7 in another region where you hold the same quota.
