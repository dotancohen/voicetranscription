//! Language code normalization for transcription providers.
//!
//! Different transcription providers expect different language code formats:
//! - Local Whisper: ISO 639-1 (e.g., "he", "en")
//! - SpeechText.AI: BCP-47 (e.g., "he-IL", "en-US")
//! - Google Cloud: BCP-47 (e.g., "he-IL", "en-US")
//! - AssemblyAI: ISO 639-1 or BCP-47
//!
//! This module provides normalization to convert user-provided language codes
//! to the format expected by each provider.

use std::collections::HashMap;
use std::sync::LazyLock;

/// Provider identifiers for language code normalization.
pub const PROVIDER_LOCAL_WHISPER: &str = "local_whisper";
pub const PROVIDER_SPEECHTEXT_AI: &str = "speechtext_ai";
pub const PROVIDER_GOOGLE_CLOUD: &str = "google_cloud";
pub const PROVIDER_ASSEMBLYAI: &str = "assemblyai";

/// Language code format expected by each provider.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LanguageFormat {
    /// ISO 639-1 two-letter codes (e.g., "he", "en")
    Iso639_1,
    /// BCP-47 language tags (e.g., "he-IL", "en-US")
    Bcp47,
}

/// Mapping of ISO 639-1 codes to default BCP-47 tags.
/// Uses the most common/default region for each language.
static ISO_TO_BCP47: LazyLock<HashMap<&'static str, &'static str>> = LazyLock::new(|| {
    HashMap::from([
        // Major languages
        ("en", "en-US"),
        ("he", "he-IL"),
        ("ar", "ar-SA"),
        ("zh", "zh-CN"),
        ("es", "es-ES"),
        ("fr", "fr-FR"),
        ("de", "de-DE"),
        ("it", "it-IT"),
        ("ja", "ja-JP"),
        ("ko", "ko-KR"),
        ("pt", "pt-BR"),
        ("ru", "ru-RU"),
        ("nl", "nl-NL"),
        ("pl", "pl-PL"),
        ("tr", "tr-TR"),
        ("vi", "vi-VN"),
        ("th", "th-TH"),
        ("id", "id-ID"),
        ("ms", "ms-MY"),
        ("hi", "hi-IN"),
        ("bn", "bn-IN"),
        ("ta", "ta-IN"),
        ("te", "te-IN"),
        ("mr", "mr-IN"),
        ("gu", "gu-IN"),
        ("kn", "kn-IN"),
        ("ml", "ml-IN"),
        ("pa", "pa-IN"),
        ("ur", "ur-PK"),
        ("fa", "fa-IR"),
        ("uk", "uk-UA"),
        ("cs", "cs-CZ"),
        ("sk", "sk-SK"),
        ("hu", "hu-HU"),
        ("ro", "ro-RO"),
        ("bg", "bg-BG"),
        ("hr", "hr-HR"),
        ("sr", "sr-RS"),
        ("sl", "sl-SI"),
        ("el", "el-GR"),
        ("sv", "sv-SE"),
        ("da", "da-DK"),
        ("fi", "fi-FI"),
        ("no", "nb-NO"),
        ("nb", "nb-NO"),
        ("nn", "nn-NO"),
        ("is", "is-IS"),
        ("et", "et-EE"),
        ("lv", "lv-LV"),
        ("lt", "lt-LT"),
        ("ca", "ca-ES"),
        ("eu", "eu-ES"),
        ("gl", "gl-ES"),
        ("cy", "cy-GB"),
        ("ga", "ga-IE"),
        ("mt", "mt-MT"),
        ("sq", "sq-AL"),
        ("mk", "mk-MK"),
        ("bs", "bs-BA"),
        ("af", "af-ZA"),
        ("sw", "sw-KE"),
        ("am", "am-ET"),
        ("yi", "yi-001"),
    ])
});

/// Mapping of BCP-47 tags to ISO 639-1 codes.
static BCP47_TO_ISO: LazyLock<HashMap<&'static str, &'static str>> = LazyLock::new(|| {
    // Build reverse mapping from ISO_TO_BCP47
    let mut map = HashMap::new();
    for (iso, bcp47) in ISO_TO_BCP47.iter() {
        map.insert(*bcp47, *iso);
    }
    // Add additional BCP-47 variants that map to the same ISO code
    map.insert("en-GB", "en");
    map.insert("en-AU", "en");
    map.insert("en-CA", "en");
    map.insert("en-IN", "en");
    map.insert("es-MX", "es");
    map.insert("es-AR", "es");
    map.insert("pt-PT", "pt");
    map.insert("zh-TW", "zh");
    map.insert("zh-HK", "zh");
    map.insert("fr-CA", "fr");
    map.insert("fr-BE", "fr");
    map.insert("de-AT", "de");
    map.insert("de-CH", "de");
    map.insert("ar-EG", "ar");
    map.insert("ar-AE", "ar");
    map.insert("ar-MA", "ar");
    map
});

/// Provider-specific language format requirements.
static PROVIDER_FORMATS: LazyLock<HashMap<&'static str, LanguageFormat>> = LazyLock::new(|| {
    HashMap::from([
        (PROVIDER_LOCAL_WHISPER, LanguageFormat::Iso639_1),
        (PROVIDER_SPEECHTEXT_AI, LanguageFormat::Bcp47),
        (PROVIDER_GOOGLE_CLOUD, LanguageFormat::Bcp47),
        (PROVIDER_ASSEMBLYAI, LanguageFormat::Iso639_1), // AssemblyAI accepts both, prefer short
    ])
});

/// Normalize a language code for a specific provider.
///
/// # Arguments
/// * `code` - The input language code (ISO 639-1 or BCP-47)
/// * `provider` - The provider identifier (e.g., "speechtext_ai")
///
/// # Returns
/// The normalized language code in the format expected by the provider.
///
/// # Examples
/// ```
/// use voice_transcription::language::normalize_language_code;
///
/// // ISO to BCP-47 for SpeechText.AI
/// assert_eq!(normalize_language_code("he", "speechtext_ai"), "he-IL");
/// assert_eq!(normalize_language_code("en", "speechtext_ai"), "en-US");
///
/// // BCP-47 to ISO for Whisper
/// assert_eq!(normalize_language_code("he-IL", "local_whisper"), "he");
///
/// // Already correct format passes through
/// assert_eq!(normalize_language_code("he-IL", "speechtext_ai"), "he-IL");
/// ```
pub fn normalize_language_code(code: &str, provider: &str) -> String {
    let code = code.trim();

    // Get the expected format for this provider (default to BCP-47)
    let expected_format = PROVIDER_FORMATS
        .get(provider)
        .copied()
        .unwrap_or(LanguageFormat::Bcp47);

    // Detect input format
    let is_bcp47 = code.contains('-');

    match (is_bcp47, expected_format) {
        // Already in expected format
        (true, LanguageFormat::Bcp47) | (false, LanguageFormat::Iso639_1) => {
            // Validate/normalize case
            if is_bcp47 {
                normalize_bcp47_case(code)
            } else {
                code.to_lowercase()
            }
        }
        // Need to convert ISO to BCP-47
        (false, LanguageFormat::Bcp47) => {
            let iso = code.to_lowercase();
            ISO_TO_BCP47
                .get(iso.as_str())
                .map(|s| s.to_string())
                .unwrap_or_else(|| {
                    // Unknown code - try constructing a reasonable BCP-47 tag
                    // by duplicating the code (e.g., "xx" -> "xx-XX")
                    format!("{}-{}", iso, iso.to_uppercase())
                })
        }
        // Need to convert BCP-47 to ISO
        (true, LanguageFormat::Iso639_1) => {
            let normalized = normalize_bcp47_case(code);
            BCP47_TO_ISO
                .get(normalized.as_str())
                .map(|s| s.to_string())
                .unwrap_or_else(|| {
                    // Extract ISO part from BCP-47 tag
                    code.split('-').next().unwrap_or(code).to_lowercase()
                })
        }
    }
}

/// Normalize BCP-47 tag case (language lowercase, region uppercase).
fn normalize_bcp47_case(code: &str) -> String {
    let parts: Vec<&str> = code.split('-').collect();
    match parts.as_slice() {
        [lang] => lang.to_lowercase(),
        [lang, region] => format!("{}-{}", lang.to_lowercase(), region.to_uppercase()),
        [lang, script, region] => format!(
            "{}-{}-{}",
            lang.to_lowercase(),
            capitalize(script),
            region.to_uppercase()
        ),
        _ => code.to_string(),
    }
}

/// Capitalize first letter (for script codes like "Hans", "Hant").
fn capitalize(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        None => String::new(),
        Some(first) => first.to_uppercase().chain(chars.flat_map(|c| c.to_lowercase())).collect(),
    }
}

/// Get the default language code for a provider.
///
/// Returns Hebrew as the default (per project requirements).
pub fn default_language_code(provider: &str) -> &'static str {
    let format = PROVIDER_FORMATS
        .get(provider)
        .copied()
        .unwrap_or(LanguageFormat::Bcp47);

    match format {
        LanguageFormat::Iso639_1 => "he",
        LanguageFormat::Bcp47 => "he-IL",
    }
}

/// Check if a language code is supported by checking if it's in our mappings.
pub fn is_known_language_code(code: &str) -> bool {
    let code_lower = code.to_lowercase();
    if code.contains('-') {
        BCP47_TO_ISO.contains_key(normalize_bcp47_case(code).as_str())
    } else {
        ISO_TO_BCP47.contains_key(code_lower.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_iso_to_bcp47_hebrew() {
        assert_eq!(normalize_language_code("he", "speechtext_ai"), "he-IL");
        assert_eq!(normalize_language_code("he", "google_cloud"), "he-IL");
    }

    #[test]
    fn test_iso_to_bcp47_english() {
        assert_eq!(normalize_language_code("en", "speechtext_ai"), "en-US");
        assert_eq!(normalize_language_code("en", "google_cloud"), "en-US");
    }

    #[test]
    fn test_bcp47_to_iso_hebrew() {
        assert_eq!(normalize_language_code("he-IL", "local_whisper"), "he");
        assert_eq!(normalize_language_code("he-IL", "assemblyai"), "he");
    }

    #[test]
    fn test_bcp47_to_iso_english_variants() {
        assert_eq!(normalize_language_code("en-US", "local_whisper"), "en");
        assert_eq!(normalize_language_code("en-GB", "local_whisper"), "en");
        assert_eq!(normalize_language_code("en-AU", "local_whisper"), "en");
    }

    #[test]
    fn test_passthrough_correct_format() {
        // BCP-47 to BCP-47 provider
        assert_eq!(normalize_language_code("he-IL", "speechtext_ai"), "he-IL");
        // ISO to ISO provider
        assert_eq!(normalize_language_code("he", "local_whisper"), "he");
    }

    #[test]
    fn test_case_normalization() {
        assert_eq!(normalize_language_code("HE", "local_whisper"), "he");
        assert_eq!(normalize_language_code("HE-il", "speechtext_ai"), "he-IL");
        assert_eq!(normalize_language_code("en-us", "speechtext_ai"), "en-US");
    }

    #[test]
    fn test_default_language_code() {
        assert_eq!(default_language_code("local_whisper"), "he");
        assert_eq!(default_language_code("speechtext_ai"), "he-IL");
        assert_eq!(default_language_code("google_cloud"), "he-IL");
    }

    #[test]
    fn test_unknown_code_iso_to_bcp47() {
        // Unknown codes get a constructed BCP-47 tag
        assert_eq!(normalize_language_code("xx", "speechtext_ai"), "xx-XX");
    }

    #[test]
    fn test_unknown_code_bcp47_to_iso() {
        // Unknown BCP-47 tags extract the language part
        assert_eq!(normalize_language_code("xx-YY", "local_whisper"), "xx");
    }

    #[test]
    fn test_is_known_language_code() {
        assert!(is_known_language_code("he"));
        assert!(is_known_language_code("en"));
        assert!(is_known_language_code("he-IL"));
        assert!(is_known_language_code("en-US"));
        assert!(!is_known_language_code("xx"));
        assert!(!is_known_language_code("xx-YY"));
    }

    #[test]
    fn test_arabic_variants() {
        assert_eq!(normalize_language_code("ar", "speechtext_ai"), "ar-SA");
        assert_eq!(normalize_language_code("ar-EG", "local_whisper"), "ar");
    }

    #[test]
    fn test_chinese_variants() {
        assert_eq!(normalize_language_code("zh", "speechtext_ai"), "zh-CN");
        assert_eq!(normalize_language_code("zh-TW", "local_whisper"), "zh");
    }
}
