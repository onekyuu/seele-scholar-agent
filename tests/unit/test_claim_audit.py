from seele_scholar_agent.nodes.claim_audit import RuleBasedClaimExtractor


def test_claim_extractor_marks_cited_sentences_as_factual():
    extractor = RuleBasedClaimExtractor()

    claims = extractor.extract_factual_claims(
        "This section introduces the topic. Prior work shows better accuracy [1]."
    )

    assert len(claims) == 1
    assert claims[0].text == "Prior work shows better accuracy [1]."
    assert claims[0].citation_numbers == (1,)


def test_claim_extractor_finds_uncited_factual_claims():
    extractor = RuleBasedClaimExtractor()

    claims = extractor.extract_factual_claims(
        "The model improves accuracy by 12%. This section explains the setup."
    )

    assert len(claims) == 1
    assert claims[0].text == "The model improves accuracy by 12%."
    assert claims[0].citation_numbers == ()


def test_claim_extractor_skips_rhetorical_and_placeholder_text():
    extractor = RuleBasedClaimExtractor()

    claims = extractor.extract_factual_claims(
        "This section discusses the motivation.\n"
        "{{FIGURE: Accuracy chart | chunks:[c1]}}\n"
        "What should future work address?"
    )

    assert claims == []
