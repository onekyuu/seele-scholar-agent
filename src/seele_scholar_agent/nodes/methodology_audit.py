import re
from dataclasses import dataclass
from typing import Literal

_NUMBER_RE = re.compile(r"\b(?:n\s*=\s*)?\d{1,6}\b", re.IGNORECASE)
_P_VALUE_RE = re.compile(r"\bp\s*[<=>]\s*0?\.\d+\b", re.IGNORECASE)
_CI_RE = re.compile(r"\b(?:ci|confidence interval|置信区间)\b", re.IGNORECASE)
_VARIANCE_RE = re.compile(
    r"\b(?:std|standard deviation|standard error|error bar|variance|方差|标准差|误差)\b",
    re.IGNORECASE,
)

_EMPIRICAL_SECTION_MARKERS = (
    "method",
    "methods",
    "methodology",
    "experiment",
    "experiments",
    "evaluation",
    "result",
    "results",
    "analysis",
    "study design",
    "方法",
    "实验",
    "评估",
    "评价",
    "结果",
    "分析",
)
_SURVEY_MARKERS = (
    "survey",
    "questionnaire",
    "respondent",
    "participant",
    "interview",
    "literature review",
    "systematic review",
    "问卷",
    "访谈",
    "受访者",
    "参与者",
    "调研",
    "综述",
)
_SAMPLE_MARKERS = (
    "sample",
    "samples",
    "participant",
    "participants",
    "respondent",
    "respondents",
    "subject",
    "subjects",
    "dataset",
    "datasets",
    "study",
    "studies",
    "paper",
    "papers",
    "样本",
    "受试者",
    "参与者",
    "受访者",
    "数据集",
    "文献",
)
_BASELINE_MARKERS = (
    "baseline",
    "baselines",
    "comparison",
    "compare",
    "compared",
    "control",
    "benchmark",
    "sota",
    "state-of-the-art",
    "基线",
    "对照",
    "比较",
    "对比",
)
_PERFORMANCE_CLAIM_MARKERS = (
    "outperform",
    "outperforms",
    "better than",
    "improves",
    "improved",
    "higher than",
    "lower than",
    "优于",
    "超过",
    "提升",
    "提高",
    "降低",
)
_METRIC_MARKERS = (
    "accuracy",
    "precision",
    "recall",
    "f1",
    "auc",
    "rmse",
    "mae",
    "latency",
    "throughput",
    "score",
    "rate",
    "准确率",
    "精确率",
    "召回率",
    "指标",
    "得分",
)
_METRIC_DEFINITION_MARKERS = (
    "defined as",
    "computed as",
    "calculated as",
    "formula",
    "definition",
    "we define",
    "定义为",
    "计算为",
    "公式",
)
_CORRELATION_MARKERS = ("correlation", "correlates", "associated with", "相关")
_CAUSAL_MARKERS = (
    "cause",
    "causes",
    "caused",
    "lead to",
    "leads to",
    "result in",
    "results in",
    "due to",
    "because of",
    "drives",
    "导致",
    "造成",
    "因为",
    "由于",
)
_SIGNIFICANCE_CLAIM_MARKERS = (
    "significant",
    "significantly",
    "statistically",
    "outperform",
    "outperforms",
    "improves",
    "improved",
    "显著",
    "明显",
    "优于",
    "提升",
)
_BACKGROUND_CLAIM_MARKERS = ("prior work", "previous studies", "existing literature", "相关工作")
_AUTHOR_RESULT_MARKERS = ("our ", "we ", "this study", "the experiment", "实验", "本文")
_PROPOSAL_RESULT_MARKERS = (
    "results show",
    "result shows",
    "experiment showed",
    "experiments showed",
    "we found",
    "we achieved",
    "achieved",
    "outperformed",
    "improved by",
    "結果は",
    "結果が",
    "実験結果",
    "示した",
    "達成した",
    "上回った",
    "改善した",
    "结果显示",
    "实验表明",
    "我们发现",
)


@dataclass(frozen=True)
class MethodologyAuditFinding:
    code: str
    review_type: Literal["weak_argument", "factual_error", "other"]
    description: str
    suggestion: str
    location: str


class MethodologyAudit:
    """Deterministic methodology/statistics checks for empirical and survey sections."""

    def audit(
        self,
        *,
        section_title: str,
        content: str,
        paper_type: str = "",
        structure_pattern: str = "",
        document_type: str = "",
    ) -> list[MethodologyAuditFinding]:
        if self._is_proposal_without_completed_results(content, document_type):
            return []
        if not self._should_audit(section_title, content, paper_type, structure_pattern):
            return []

        text = f"{section_title}\n{content}"
        lowered = text.casefold()
        findings: list[MethodologyAuditFinding] = []

        if self._mentions_sample(lowered) and not self._has_sample_size(text):
            findings.append(
                MethodologyAuditFinding(
                    code="METHODOLOGY_SAMPLE_SIZE_MISSING",
                    review_type="weak_argument",
                    description="Sample size or study count is not clearly reported.",
                    suggestion=(
                        "Report the sample size, dataset size, participant count, "
                        "or included study count."
                    ),
                    location=section_title,
                )
            )

        if self._makes_performance_claim(lowered) and not self._mentions_baseline(lowered):
            findings.append(
                MethodologyAuditFinding(
                    code="METHODOLOGY_BASELINE_FAIRNESS_MISSING",
                    review_type="weak_argument",
                    description=(
                        "Performance comparison lacks a clear baseline or control condition."
                    ),
                    suggestion=(
                        "Specify comparable baselines, controls, or benchmark settings "
                        "and explain fairness."
                    ),
                    location=section_title,
                )
            )

        if self._mentions_metric(lowered) and not self._defines_metric(lowered):
            findings.append(
                MethodologyAuditFinding(
                    code="METHODOLOGY_METRIC_DEFINITION_MISSING",
                    review_type="weak_argument",
                    description="Evaluation metrics are named but not clearly defined.",
                    suggestion=(
                        "Define each metric, calculation method, and direction of improvement."
                    ),
                    location=section_title,
                )
            )

        if self._mixes_correlation_and_causation(lowered):
            findings.append(
                MethodologyAuditFinding(
                    code="METHODOLOGY_CORRELATION_CAUSATION_MIXED",
                    review_type="factual_error",
                    description="Correlation language is mixed with causal claims.",
                    suggestion=(
                        "Use associative language or justify causal identification "
                        "with design assumptions."
                    ),
                    location=section_title,
                )
            )

        if self._needs_uncertainty_report(lowered) and not self._reports_uncertainty(text):
            findings.append(
                MethodologyAuditFinding(
                    code="METHODOLOGY_SIGNIFICANCE_UNCERTAINTY_MISSING",
                    review_type="weak_argument",
                    description=(
                        "Comparative or significant results lack significance tests "
                        "or uncertainty intervals."
                    ),
                    suggestion=(
                        "Add p-values, confidence intervals, variance estimates, "
                        "or explain why they are not needed."
                    ),
                    location=section_title,
                )
            )

        return findings

    def _is_proposal_without_completed_results(
        self, content: str, document_type: str
    ) -> bool:
        normalized_type = document_type.casefold().replace("-", "_").replace(" ", "_")
        if "research_proposal" not in normalized_type and normalized_type not in {
            "proposal",
            "研究計画書",
        }:
            return False
        lowered = content.casefold()
        return not any(marker.casefold() in lowered for marker in _PROPOSAL_RESULT_MARKERS)

    def _should_audit(
        self, section_title: str, content: str, paper_type: str, structure_pattern: str
    ) -> bool:
        type_text = f"{paper_type} {structure_pattern}".casefold()
        title_text = section_title.casefold()
        content_text = content.casefold()
        if any(marker in type_text for marker in (*_EMPIRICAL_SECTION_MARKERS, *_SURVEY_MARKERS)):
            return True
        if any(
            marker in content_text for marker in (*_EMPIRICAL_SECTION_MARKERS, *_SURVEY_MARKERS)
        ):
            return True
        if any(
            marker in title_text
            for marker in (
                "method",
                "methodology",
                "experiment",
                "evaluation",
                "survey",
                "方法",
                "实验",
                "调研",
            )
        ):
            return True
        title_has_results = any(
            marker in title_text for marker in ("result", "analysis", "结果", "分析")
        )
        content_has_statistical_claim = any(
            marker in content_text
            for marker in (
                *_SAMPLE_MARKERS,
                *_PERFORMANCE_CLAIM_MARKERS,
                *_METRIC_MARKERS,
                *_CORRELATION_MARKERS,
                *_SIGNIFICANCE_CLAIM_MARKERS,
            )
        )
        return title_has_results and content_has_statistical_claim

    def _mentions_sample(self, lowered: str) -> bool:
        return any(marker in lowered for marker in _SAMPLE_MARKERS)

    def _has_sample_size(self, text: str) -> bool:
        return bool(_NUMBER_RE.search(text))

    def _makes_performance_claim(self, lowered: str) -> bool:
        if self._is_background_claim(lowered):
            return False
        return any(marker in lowered for marker in _PERFORMANCE_CLAIM_MARKERS)

    def _mentions_baseline(self, lowered: str) -> bool:
        return any(marker in lowered for marker in _BASELINE_MARKERS)

    def _mentions_metric(self, lowered: str) -> bool:
        return any(marker in lowered for marker in _METRIC_MARKERS)

    def _defines_metric(self, lowered: str) -> bool:
        return any(marker in lowered for marker in _METRIC_DEFINITION_MARKERS)

    def _mixes_correlation_and_causation(self, lowered: str) -> bool:
        return any(marker in lowered for marker in _CORRELATION_MARKERS) and any(
            marker in lowered for marker in _CAUSAL_MARKERS
        )

    def _needs_uncertainty_report(self, lowered: str) -> bool:
        if self._is_background_claim(lowered):
            return False
        return any(marker in lowered for marker in _SIGNIFICANCE_CLAIM_MARKERS)

    def _reports_uncertainty(self, text: str) -> bool:
        return bool(_P_VALUE_RE.search(text) or _CI_RE.search(text) or _VARIANCE_RE.search(text))

    def _is_background_claim(self, lowered: str) -> bool:
        return any(marker in lowered for marker in _BACKGROUND_CLAIM_MARKERS) and not any(
            marker in lowered for marker in _AUTHOR_RESULT_MARKERS
        )
