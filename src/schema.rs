//! Provider options schema for UI generation.
//!
//! This module defines types that describe configurable options for each
//! transcription provider. The UI uses these schemas to dynamically generate
//! configuration forms.

use serde::{Deserialize, Serialize};

/// A value option for Select-type options.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptionValue {
    /// The actual value to use in configuration.
    pub value: String,
    /// Human-readable label for display.
    pub label: String,
}

impl OptionValue {
    /// Create a new option value.
    pub fn new(value: impl Into<String>, label: impl Into<String>) -> Self {
        Self {
            value: value.into(),
            label: label.into(),
        }
    }

    /// Create an option value where value and label are the same.
    pub fn simple(value: impl Into<String>) -> Self {
        let v = value.into();
        Self {
            label: v.clone(),
            value: v,
        }
    }
}

/// The type of input control for an option.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OptionType {
    /// Single-line text input.
    Text,
    /// Numeric input (integer or float).
    Number,
    /// Dropdown selection from predefined values.
    Select,
    /// Boolean checkbox.
    Checkbox,
    /// File path input with browse button.
    Path,
}

/// A configurable option for a transcription provider.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderOption {
    /// Unique identifier for this option (used as config key).
    pub id: String,
    /// Human-readable label for the option.
    pub label: String,
    /// The type of input control to use.
    pub option_type: OptionType,
    /// Whether this option must be provided.
    pub required: bool,
    /// Default value as a string (empty if no default).
    pub default: String,
    /// Available values for Select type options.
    pub values: Vec<OptionValue>,
    /// Optional description/help text.
    pub description: Option<String>,
}

impl ProviderOption {
    /// Create a new provider option.
    pub fn new(id: impl Into<String>, label: impl Into<String>, option_type: OptionType) -> Self {
        Self {
            id: id.into(),
            label: label.into(),
            option_type,
            required: false,
            default: String::new(),
            values: Vec::new(),
            description: None,
        }
    }

    /// Mark this option as required.
    pub fn required(mut self) -> Self {
        self.required = true;
        self
    }

    /// Set the default value.
    pub fn with_default(mut self, default: impl Into<String>) -> Self {
        self.default = default.into();
        self
    }

    /// Set available values (for Select type).
    pub fn with_values(mut self, values: Vec<OptionValue>) -> Self {
        self.values = values;
        self
    }

    /// Set the description.
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }
}

/// Schema describing a transcription provider's configurable options.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderSchema {
    /// Unique identifier for the provider.
    pub provider_id: String,
    /// Human-readable name for the provider.
    pub provider_name: String,
    /// List of configurable options.
    pub options: Vec<ProviderOption>,
}

impl ProviderSchema {
    /// Create a new provider schema.
    pub fn new(provider_id: impl Into<String>, provider_name: impl Into<String>) -> Self {
        Self {
            provider_id: provider_id.into(),
            provider_name: provider_name.into(),
            options: Vec::new(),
        }
    }

    /// Add an option to the schema.
    pub fn with_option(mut self, option: ProviderOption) -> Self {
        self.options.push(option);
        self
    }

    /// Add multiple options to the schema.
    pub fn with_options(mut self, options: Vec<ProviderOption>) -> Self {
        self.options.extend(options);
        self
    }
}

/// Trait for backends that can provide their configuration schema.
pub trait HasProviderSchema {
    /// Get the schema describing this provider's options.
    fn get_provider_schema() -> ProviderSchema;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_option_value_new() {
        let opt = OptionValue::new("tiny", "Tiny (75MB)");
        assert_eq!(opt.value, "tiny");
        assert_eq!(opt.label, "Tiny (75MB)");
    }

    #[test]
    fn test_option_value_simple() {
        let opt = OptionValue::simple("base");
        assert_eq!(opt.value, "base");
        assert_eq!(opt.label, "base");
    }

    #[test]
    fn test_option_type_serialization() {
        let types = vec![
            (OptionType::Text, "\"text\""),
            (OptionType::Number, "\"number\""),
            (OptionType::Select, "\"select\""),
            (OptionType::Checkbox, "\"checkbox\""),
            (OptionType::Path, "\"path\""),
        ];

        for (opt_type, expected) in types {
            let json = serde_json::to_string(&opt_type).unwrap();
            assert_eq!(json, expected);
        }
    }

    #[test]
    fn test_provider_option_builder() {
        let option = ProviderOption::new("model", "Model", OptionType::Select)
            .required()
            .with_default("base")
            .with_values(vec![
                OptionValue::new("tiny", "Tiny"),
                OptionValue::new("base", "Base"),
            ])
            .with_description("Whisper model to use");

        assert_eq!(option.id, "model");
        assert_eq!(option.label, "Model");
        assert_eq!(option.option_type, OptionType::Select);
        assert!(option.required);
        assert_eq!(option.default, "base");
        assert_eq!(option.values.len(), 2);
        assert_eq!(option.description, Some("Whisper model to use".to_string()));
    }

    #[test]
    fn test_provider_schema_builder() {
        let schema = ProviderSchema::new("local_whisper", "Local Whisper")
            .with_option(ProviderOption::new("model", "Model", OptionType::Select));

        assert_eq!(schema.provider_id, "local_whisper");
        assert_eq!(schema.provider_name, "Local Whisper");
        assert_eq!(schema.options.len(), 1);
    }

    #[test]
    fn test_provider_schema_serialization() {
        let schema = ProviderSchema::new("test_provider", "Test Provider")
            .with_option(
                ProviderOption::new("api_key", "API Key", OptionType::Text)
                    .required()
                    .with_description("Your API key"),
            );

        let json = serde_json::to_string_pretty(&schema).unwrap();
        let deserialized: ProviderSchema = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.provider_id, "test_provider");
        assert_eq!(deserialized.options.len(), 1);
        assert_eq!(deserialized.options[0].id, "api_key");
        assert!(deserialized.options[0].required);
    }
}
