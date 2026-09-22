"""Typed runtime settings, resolved from environment variables and ``.env``.

Single source of truth for every knob that is *deployment*, not *experiment*
(experiment knobs live in YAML, see :mod:`traffic_rl.config_loader`).

Conventions
-----------
* Env var names are the canonical ones documented in ``.env.example``
  (``AWS_REGION``, ``S3_ARTIFACT_BUCKET``, ``SAGEMAKER_ROLE_ARN``, ...).
* Secrets are :class:`pydantic.SecretStr`, so they never appear in
  ``repr()``/``str()`` or an accidental log line; use
  :meth:`AppSettings.redacted_dump` when you need to persist them.
* Nested groups (:attr:`AppSettings.aws`, :attr:`AppSettings.sagemaker`, ...)
  are exposed as read-only views for ergonomic access in code.
* Nothing here talks to AWS: values are *declared* here and *resolved* by
  :mod:`traffic_rl.cloud.aws.secrets` only when a real run needs them.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

Environment = Literal["local", "dev", "staging", "prod", "ci"]


class AwsSettings(BaseModel):
    """AWS account/credential configuration."""

    region: str = "us-east-1"
    account_id: str | None = None
    profile: str | None = None
    role_arn: str | None = None
    access_key_id: SecretStr | None = None
    secret_access_key: SecretStr | None = None
    session_token: SecretStr | None = None
    endpoint_url: str | None = None
    dry_run: bool = False

    @property
    def has_static_credentials(self) -> bool:
        """True when long-lived keys are present (local development only)."""
        return bool(self.access_key_id and self.secret_access_key)

    @property
    def has_identity(self) -> bool:
        """True when *some* credential source is configured."""
        return bool(self.role_arn or self.profile or self.has_static_credentials)


class S3Settings(BaseModel):
    """Artifact/data bucket layout."""

    artifact_bucket: str = "traffic-rl-artifacts"
    data_bucket: str | None = None
    scenario_bucket: str | None = None
    prefix: str = "traffic-rl"
    kms_key_id: str | None = None
    lifecycle_days: int = 90
    multipart_threshold_mb: int = 64

    def uri(self, relative_path: str) -> str:
        """Return the ``s3://`` URI for a path inside the artifact prefix."""
        key = "/".join(part.strip("/") for part in (self.prefix, relative_path) if part)
        return f"s3://{self.artifact_bucket}/{key}"


class SageMakerSettings(BaseModel):
    """SageMaker training, HPO, processing and endpoint configuration."""

    # ``model_package_group`` is a legitimate SageMaker term that collides with
    # pydantic's protected ``model_`` namespace, so opt out of that protection.
    model_config = ConfigDict(protected_namespaces=())

    role_arn: str | None = None
    instance_type: str = "ml.g5.2xlarge"
    volume_size_gb: int = 100
    use_spot: bool = True
    max_runtime_s: int = 86_400
    max_wait_s: int = 93_600
    keep_alive_period_s: int = 1800
    training_image_uri: str | None = None
    output_prefix: str = "sagemaker-output"
    processing_instance_type: str = "ml.m5.2xlarge"
    endpoint_instance_type: str = "ml.g4dn.xlarge"
    endpoint_name: str = "traffic-rl-policy"
    model_package_group: str = "traffic-rl-policies"
    hpo_max_jobs: int = 20
    hpo_max_parallel_jobs: int = 4
    hpo_strategy: Literal["Bayesian", "Random", "Hyperband", "Grid"] = "Bayesian"

    @property
    def spot_parameter(self) -> str:
        """``use_spot_instances`` value expected by the SageMaker SDK/API."""
        return "true" if self.use_spot else "false"


class TrackingSettings(BaseModel):
    """Where metrics and experiment metadata are sent."""

    sinks: str = "jsonl"
    tensorboard: bool = True
    mlflow_tracking_uri: str | None = None
    mlflow_experiment: str = "traffic-rl"
    wandb_project: str | None = None
    wandb_api_key: SecretStr | None = None
    cloudwatch_namespace: str = "TrafficRL"


class LoggingSettings(BaseModel):
    """Structured logging configuration."""

    level: str = "INFO"
    format: Literal["json", "console"] = "json"
    dir: Path = Path("logs")
    use_queue: bool = True
    max_bytes: int = 50 * 1024 * 1024
    backup_count: int = 5
    silence_noisy: bool = True


class SumoSettings(BaseModel):
    """SUMO installation and connection pool."""

    home: str | None = None
    binary: str | None = None
    gui_binary: str | None = None
    port_start: int = 8813
    port_end: int = 8900
    step_length_s: float = 1.0
    gui_delay_ms: int = 100
    extra_args: str = ""

    @property
    def extra_arg_list(self) -> list[str]:
        """Split ``SUMO_EXTRA_ARGS`` into an argument list."""
        return [part for part in self.extra_args.split() if part]


class RunSettings(BaseModel):
    """Run identity and artefact locations."""

    env: Environment = "local"
    config: str = "configs/base.yaml"
    experiment: str | None = None
    run_id: str | None = None
    runs_dir: Path = Path("runs")
    data_dir: Path = Path("data")
    seed: int = 42
    dry_run: bool = False
    max_parallel_envs: int = 4
    torch_num_threads: int = 0
    deterministic: bool = True

    @property
    def runs_root(self) -> Path:
        return Path(self.runs_dir)


class NotificationSettings(BaseModel):
    """Alerting destinations for pipeline events."""

    sns_topic_arn: str | None = None
    slack_webhook_url: SecretStr | None = None


class AppSettings(BaseSettings):
    """Top-level settings: environment variables first, ``.env`` second.

    Field names are the lower-cased env var names, so ``AWS_REGION`` maps to
    :attr:`aws_region` and ``S3_ARTIFACT_BUCKET`` to :attr:`s3_artifact_bucket`
    without any alias plumbing.  Grouped views are exposed through
    :attr:`aws`, :attr:`s3`, :attr:`sagemaker`, :attr:`sumo`, :attr:`tracking`,
    :attr:`logging`, :attr:`run` and :attr:`notifications`.
    """

    model_config = SettingsConfigDict(
        env_file=(".env", ".env.local"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # -- core run identity ------------------------------------------------------
    traffic_rl_env: Environment = Field("local", description="Deployment environment.")
    traffic_rl_config: str = Field("configs/base.yaml", description="YAML config path.")
    experiment_name: str | None = Field(None, description="Experiment/group name.")
    run_id: str | None = Field(None, description="Explicit run id (auto-generated if unset).")
    seed: int = Field(42, description="Global random seed.")
    dry_run: bool = Field(False, description="Disable every cloud side effect.")
    runs_dir: Path = Field(Path("runs"), description="Artefact root directory.")
    data_dir: Path = Field(Path("data"), description="Local dataset/scenario root.")
    max_parallel_envs: int = Field(4, description="Vectorised SUMO workers.")
    torch_num_threads: int = Field(0, description="0 = let torch decide.")
    deterministic: bool = Field(True, description="torch.use_deterministic_algorithms.")
    test_report_path: str | None = Field(None, description="TEST_REPORT_PATH for CI.")

    # -- logging -----------------------------------------------------------------
    log_level: str = Field("INFO", description="DEBUG | INFO | WARNING | ERROR.")
    log_format: Literal["json", "console"] = Field("json", description="Log record format.")
    log_dir: Path = Field(Path("logs"), description="Fallback log directory.")
    log_use_queue: bool = Field(True, description="Async logging via QueueHandler.")
    log_max_bytes: int = Field(50 * 1024 * 1024, description="Rotation size in bytes.")
    log_backup_count: int = Field(5, description="Rotated files to keep.")

    # -- SUMO --------------------------------------------------------------------
    sumo_home: str | None = Field(None, description="SUMO installation directory.")
    sumo_binary: str | None = Field(None, description="Explicit path to `sumo`.")
    sumo_gui_binary: str | None = Field(None, description="Explicit path to `sumo-gui`.")
    sumo_port_start: int = Field(8813, description="First traci remote port.")
    sumo_port_end: int = Field(8900, description="Last traci remote port.")
    sumo_extra_args: str = Field("", description="Extra SUMO CLI arguments.")

    # -- AWS identity ------------------------------------------------------------
    aws_region: str = Field("us-east-1", description="AWS region.")
    aws_account_id: str | None = Field(None, description="AWS account id.")
    aws_profile: str | None = Field(None, description="Named profile (local dev).")
    aws_role_arn: str | None = Field(None, description="Role assumed via OIDC/STS.")
    aws_access_key_id: SecretStr | None = Field(None, description="Local-only access key.")
    aws_secret_access_key: SecretStr | None = Field(None, description="Local-only secret key.")
    aws_session_token: SecretStr | None = Field(None, description="Local-only session token.")
    aws_endpoint_url: str | None = Field(None, description="Override endpoint (LocalStack).")

    # -- S3 ----------------------------------------------------------------------
    s3_artifact_bucket: str = Field("traffic-rl-artifacts", description="Artefact bucket.")
    s3_data_bucket: str | None = Field(None, description="Datasets/scenarios bucket.")
    s3_scenario_bucket: str | None = Field(None, description="Generated scenarios bucket.")
    s3_prefix: str = Field("traffic-rl", description="Key prefix for all artefacts.")
    s3_kms_key_id: str | None = Field(None, description="KMS key for SSE-KMS uploads.")
    s3_lifecycle_days: int = Field(90, description="Artefact retention in days.")
    s3_multipart_threshold_mb: int = Field(64, description="Multipart upload threshold.")

    # -- SageMaker ---------------------------------------------------------------
    sagemaker_role_arn: str | None = Field(None, description="Execution role for jobs.")
    sagemaker_instance_type: str = Field("ml.g5.2xlarge", description="Training instance.")
    sagemaker_volume_size_gb: int = Field(100, description="Training EBS volume (GiB).")
    sagemaker_use_spot: bool = Field(True, description="Managed spot training.")
    sagemaker_max_runtime_s: int = Field(86_400, description="Training timeout (seconds).")
    sagemaker_max_wait_s: int = Field(93_600, description="Spot max wait (seconds).")
    sagemaker_keep_alive_period_s: int = Field(1800, description="Warm pool reuse (seconds).")
    sagemaker_training_image_uri: str | None = Field(None, description="ECR image URI.")
    sagemaker_output_prefix: str = Field("sagemaker-output", description="S3 output prefix.")
    sagemaker_processing_instance_type: str = Field(
        "ml.m5.2xlarge", description="Evaluation processing instance."
    )
    sagemaker_endpoint_instance_type: str = Field(
        "ml.g4dn.xlarge", description="Realtime endpoint instance."
    )
    sagemaker_endpoint_name: str = Field("traffic-rl-policy", description="Endpoint name.")
    sagemaker_model_package_group: str = Field(
        "traffic-rl-policies", description="Model Registry package group."
    )
    sagemaker_hpo_max_jobs: int = Field(20, description="Tuning jobs to launch.")
    sagemaker_hpo_max_parallel_jobs: int = Field(4, description="Concurrent tuning jobs.")
    sagemaker_hpo_strategy: Literal["Bayesian", "Random", "Hyperband", "Grid"] = Field(
        "Bayesian", description="HPO search strategy."
    )

    # -- ECR ---------------------------------------------------------------------
    ecr_registry: str | None = Field(None, description="<account>.dkr.ecr.<region>.amazonaws.com")
    ecr_repository: str = Field("traffic-rl", description="ECR repository name.")
    image_tag: str = Field("latest", description="Image tag for this build.")

    # -- secret stores -----------------------------------------------------------
    ssm_prefix: str = Field("/traffic-rl", description="SSM Parameter Store prefix.")
    secrets_prefix: str = Field("traffic-rl", description="Secrets Manager name prefix.")

    # -- tracking ----------------------------------------------------------------
    tracking_sinks: str = Field("jsonl", description="Comma-separated metrics sinks.")
    tensorboard_enabled: bool = Field(True, description="Write TensorBoard events.")
    mlflow_tracking_uri: str | None = Field(None, description="MLflow tracking server.")
    mlflow_experiment: str = Field("traffic-rl", description="MLflow experiment name.")
    wandb_project: str | None = Field(None, description="Weights & Biases project.")
    wandb_api_key: SecretStr | None = Field(None, description="W&B API key.")
    cloudwatch_namespace: str = Field("TrafficRL", description="CloudWatch metric namespace.")
    cloudwatch_log_group: str = Field("/traffic-rl/{{env}}", description="CloudWatch log group.")

    # -- notifications -----------------------------------------------------------
    sns_topic_arn: str | None = Field(None, description="SNS topic for pipeline alerts.")
    slack_webhook_url: SecretStr | None = Field(None, description="Slack webhook (secret).")


    # -- validators --------------------------------------------------------------
    _OPTIONAL_STRINGS: ClassVar[tuple[str, ...]] = (
        "experiment_name", "run_id", "sumo_home", "sumo_binary", "sumo_gui_binary",
        "aws_account_id", "aws_profile", "aws_role_arn", "aws_endpoint_url",
        "aws_access_key_id", "aws_secret_access_key", "aws_session_token",
        "s3_data_bucket", "s3_scenario_bucket", "s3_kms_key_id",
        "sagemaker_role_arn", "sagemaker_training_image_uri",
        "ecr_registry", "mlflow_tracking_uri", "wandb_project", "wandb_api_key",
        "sns_topic_arn", "slack_webhook_url", "test_report_path",
    )

    @field_validator(*_OPTIONAL_STRINGS, mode="before")
    @classmethod
    def _empty_is_none(cls, value: Any) -> Any:
        return _clean_optional(value)

    @field_validator("log_level", mode="before")
    @classmethod
    def _normalise_level(cls, value: Any) -> Any:
        return str(value).strip().upper() if value is not None else "INFO"

    @field_validator("s3_prefix", "ssm_prefix", "secrets_prefix", mode="before")
    @classmethod
    def _strip_slashes(cls, value: Any) -> Any:
        return str(value).strip("/") if value is not None else value

    # -- grouped views -----------------------------------------------------------
    @property
    def aws(self) -> AwsSettings:
        return AwsSettings(
            region=self.aws_region,
            account_id=self.aws_account_id,
            profile=self.aws_profile,
            role_arn=self.aws_role_arn,
            access_key_id=self.aws_access_key_id,
            secret_access_key=self.aws_secret_access_key,
            session_token=self.aws_session_token,
            endpoint_url=self.aws_endpoint_url,
            dry_run=self.dry_run,
        )

    @property
    def s3(self) -> S3Settings:
        return S3Settings(
            artifact_bucket=self.s3_artifact_bucket,
            data_bucket=self.s3_data_bucket,
            scenario_bucket=self.s3_scenario_bucket,
            prefix=self.s3_prefix,
            kms_key_id=self.s3_kms_key_id,
            lifecycle_days=self.s3_lifecycle_days,
            multipart_threshold_mb=self.s3_multipart_threshold_mb,
        )

    @property
    def sagemaker(self) -> SageMakerSettings:
        return SageMakerSettings(
            role_arn=self.sagemaker_role_arn,
            instance_type=self.sagemaker_instance_type,
            volume_size_gb=self.sagemaker_volume_size_gb,
            use_spot=self.sagemaker_use_spot,
            max_runtime_s=self.sagemaker_max_runtime_s,
            max_wait_s=self.sagemaker_max_wait_s,
            keep_alive_period_s=self.sagemaker_keep_alive_period_s,
            training_image_uri=self.sagemaker_training_image_uri,
            output_prefix=self.sagemaker_output_prefix,
            processing_instance_type=self.sagemaker_processing_instance_type,
            endpoint_instance_type=self.sagemaker_endpoint_instance_type,
            endpoint_name=self.sagemaker_endpoint_name,
            model_package_group=self.sagemaker_model_package_group,
            hpo_max_jobs=self.sagemaker_hpo_max_jobs,
            hpo_max_parallel_jobs=self.sagemaker_hpo_max_parallel_jobs,
            hpo_strategy=self.sagemaker_hpo_strategy,
        )

    @property
    def sumo(self) -> SumoSettings:
        return SumoSettings(
            home=self.sumo_home,
            binary=self.sumo_binary,
            gui_binary=self.sumo_gui_binary,
            port_start=self.sumo_port_start,
            port_end=self.sumo_port_end,
            extra_args=self.sumo_extra_args,
        )

    @property
    def tracking(self) -> TrackingSettings:
        return TrackingSettings(
            sinks=self.tracking_sinks,
            tensorboard=self.tensorboard_enabled,
            mlflow_tracking_uri=self.mlflow_tracking_uri,
            mlflow_experiment=self.mlflow_experiment,
            wandb_project=self.wandb_project,
            wandb_api_key=self.wandb_api_key,
            cloudwatch_namespace=self.cloudwatch_namespace,
        )

    @property
    def log(self) -> LoggingSettings:
        return LoggingSettings(
            level=self.log_level,
            format=self.log_format,
            dir=self.log_dir,
            use_queue=self.log_use_queue,
            max_bytes=self.log_max_bytes,
            backup_count=self.log_backup_count,
        )

    @property
    def run_settings(self) -> RunSettings:
        return RunSettings(
            env=self.traffic_rl_env,
            config=self.traffic_rl_config,
            experiment=self.experiment_name,
            run_id=self.run_id,
            runs_dir=self.runs_dir,
            data_dir=self.data_dir,
            seed=self.seed,
            dry_run=self.dry_run,
            max_parallel_envs=self.max_parallel_envs,
            torch_num_threads=self.torch_num_threads,
            deterministic=self.deterministic,
        )

    @property
    def notifications(self) -> NotificationSettings:
        return NotificationSettings(
            sns_topic_arn=self.sns_topic_arn,
            slack_webhook_url=self.slack_webhook_url,
        )

    @property
    def log_group(self) -> str:
        """CloudWatch log group with ``{{env}}`` resolved."""
        return self.cloudwatch_log_group.replace("{{env}}", self.traffic_rl_env)

    @property
    def image_uri(self) -> str | None:
        """Full ECR image URI for this build (``None`` when not configured)."""
        if self.sagemaker_training_image_uri:
            return self.sagemaker_training_image_uri
        if self.ecr_registry:
            return f"{self.ecr_registry}/{self.ecr_repository}:{self.image_tag}"
        return None

    # -- helpers -----------------------------------------------------------------
    def secret_values(self) -> list[str]:
        """Return *plaintext* secrets for the log redaction filter.

        Only call this while configuring logging; the result must never be
        logged, serialised or uploaded.
        """
        values: list[str] = []
        for field_name in self._OPTIONAL_STRINGS:
            if field_name.endswith(("_key", "_token", "webhook_url")):
                field = getattr(self, field_name, None)
                if isinstance(field, SecretStr):
                    values.append(field.get_secret_value())
        return [value for value in values if value]

    def redacted_dump(self) -> dict[str, Any]:
        """Return a JSON-serialisable dump with secrets masked.

        Safe to persist as ``runs/<run_id>/config.json``: it documents exactly
        which infrastructure a run used without leaking credentials.
        """
        return self.model_dump(mode="json")

    def apply_to_environment(self) -> None:
        """Export the settings third-party libraries read from ``os.environ``.

        ``traci``/SUMO read ``SUMO_HOME`` themselves and botocore honours the
        ``AWS_*`` variables, so propagating them keeps a run behaving the same
        whether it was started by the CLI, a container or SageMaker.
        """
        mapping = {
            "SUMO_HOME": self.sumo_home,
            "SUMO_BINARY": self.sumo_binary,
            "SUMO_GUI_BINARY": self.sumo_gui_binary,
            "AWS_REGION": self.aws_region,
            "AWS_DEFAULT_REGION": self.aws_region,
            "AWS_PROFILE": self.aws_profile,
            "TRAFFIC_RL_ENV": self.traffic_rl_env,
        }
        for key, value in mapping.items():
            if value:
                os.environ.setdefault(key, str(value))
        if self.aws_access_key_id:
            os.environ.setdefault("AWS_ACCESS_KEY_ID", self.aws_access_key_id.get_secret_value())
        if self.aws_secret_access_key:
            os.environ.setdefault(
                "AWS_SECRET_ACCESS_KEY", self.aws_secret_access_key.get_secret_value()
            )
        if self.aws_session_token:
            os.environ.setdefault("AWS_SESSION_TOKEN", self.aws_session_token.get_secret_value())
        if self.torch_num_threads:
            os.environ.setdefault("OMP_NUM_THREADS", str(self.torch_num_threads))


def _clean_optional(value: Any) -> Any:
    """Treat empty strings from ``.env`` as *unset*."""
    if isinstance(value, str) and not value.strip():
        return None
    return value


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """Return the process-wide settings singleton.

    Cached because parsing ``.env`` and building the model is comparatively
    expensive and settings are immutable for the lifetime of a run.  Tests and
    entry points that change the environment call
    :func:`reset_settings_cache` first.
    """
    settings = AppSettings()
    settings.apply_to_environment()
    return settings


def reset_settings_cache() -> None:
    """Drop the cached settings so the next call re-reads the environment."""
    get_settings.cache_clear()


def describe_settings() -> list[dict[str, Any]]:
    """Return documentation-ready metadata for every settings field.

    Used by ``scripts/gen_config_docs.py`` so ``docs/configuration.md`` is
    generated from the code and can never drift from it.
    """
    rows: list[dict[str, Any]] = []
    for name, field in AppSettings.model_fields.items():
        annotation = str(field.annotation).replace("typing.", "").replace("pydantic.", "")
        default = field.default
        rows.append(
            {
                "env_var": name.upper(),
                "field": name,
                "type": annotation,
                "default": "" if default is None else str(default),
                "required": field.is_required(),
                "description": field.description or "",
                "secret": "SecretStr" in annotation,
            }
        )
    return sorted(rows, key=lambda row: row["env_var"])
