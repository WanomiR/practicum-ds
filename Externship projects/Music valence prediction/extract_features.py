def get_features(file_path, duration=25):
    """
    Extracts features from an audio file.
    """

    import librosa
    import get_stats
    from numpy impor concatenate

    y, sr = librosa.load(file_path, duration=duration)

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=24)
    mfccs_feature = get_stats(mfccs)

    cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    cent_feature = get_stats(cent)

    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.95)
    rolloff_middle = librosa.feature.spectral_rolloff(
        y=y, sr=sr, roll_percent=0.5)
    rolloff_min = librosa.feature.spectral_rolloff(
        y=y, sr=sr, roll_percent=0.01)
    rolloff_feature = np.concatenate(
        (get_stats(rolloff), get_stats(rolloff_middle), get_stats(rolloff_min))
    )

    contrast = librosa.feature.spectral_contrast(
        S=np.abs(librosa.stft(y=y)), sr=sr)
    contrast_feature = get_stats(contrast)

    chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr)
    chroma_cens_feature = get_stats(chroma_cens)

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    dtempo = librosa.beat.tempo(
        onset_envelope=onset_env, sr=sr, aggregate=None)
    dtempo_feature = get_stats(dtempo)

    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_feature = get_stats(zcr)

    rms = librosa.feature.rms(y=y)[0]
    rms_feature = get_stats(rms)

    features = np.concatenate(
        (
            mfccs_feature,
            cent_feature,
            rolloff_feature,
            contrast_feature,
            chroma_cens_feature,
            dtempo_feature,
            zcr_feature,
            rms_feature,
        )
    )

    return features
