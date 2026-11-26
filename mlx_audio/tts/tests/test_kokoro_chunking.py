from mlx_audio.tts.models.kokoro.pipeline import KokoroPipeline


def quiet_pipeline(lang_code):
    pipeline = KokoroPipeline.__new__(KokoroPipeline)
    pipeline.lang_code = lang_code
    pipeline.model = False
    pipeline.voices = {}
    # One phoneme per character, like espeak's rough ratio for Spanish.
    pipeline.g2p = lambda text: (text, None)
    return pipeline


def test_long_sentence_without_end_punctuation_is_not_truncated():
    text = "una palabra tras otra sin ningún punto, " * 40

    results = list(quiet_pipeline("e")(text, voice="ef_dora"))

    assert all(len(result.graphemes) <= 400 for result in results)
    spoken = " ".join(result.phonemes for result in results)
    assert spoken.split() == text.split()


def test_short_sentences_are_still_grouped_into_chunks():
    text = "Esta es una frase de prueba. " * 20

    results = list(quiet_pipeline("e")(text, voice="ef_dora"))

    assert [len(result.graphemes) for result in results] == [376, 202]
