# On-demand list prices, fetched 2026-09-30

AWS: Price List API (pricing GetProducts), US East (Ohio).
GCP: Cloud Billing Catalog API, services 6F81-5844-456A (Compute Engine) and
152E-C115-5142 (Cloud Run), OnDemand SKUs. VM price = GPU + vCPU + RAM SKUs.

| Option | $/hour |
| --- | ---: |
| SageMaker ml.g4dn.xlarge (1x T4) | 0.7360 |
| SageMaker ml.g6.xlarge (1x L4) | 1.1267 |
| EC2 g4dn.xlarge (1x T4) | 0.5260 |
| EC2 g6.xlarge (1x L4) | 0.8048 |
| GCE n1-standard-2 + T4, us-west2 (gpu-vllm-t4-2b rig) | 0.5241 |
| GCE n1-standard-2 + T4, us-east4 | 0.4770 |
| GCE g2-standard-4 (1x L4, 4 vCPU, 16 GB), us-east4 | 0.7045 |
| GCE g2-standard-8 (1x L4, 8 vCPU, 32 GB), us-east4 | 0.8508 |
| Cloud Run 1x L4, 8 vCPU, 32 GiB, instance-based, no zonal redundancy, us-east4 (gpu-2B-cloudrun-devops-agent settings) | 1.4209 |
| Cloud Run, same, with zonal redundancy | 1.7960 |

Cloud Run unit prices, us-east4: CPU 1.8e-05 $/vCPU-s, memory 2e-06 $/GiB-s,
L4 0.0001867 $/s (no zonal redundancy), 0.0002909 $/s (zonal redundancy).
