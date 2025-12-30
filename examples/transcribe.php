#!/usr/bin/env php
<?php
/**
 * Simple example of using VoiceTranscription in PHP.
 *
 * This example uses a Python subprocess since native PHP bindings
 * are not yet available. For production use, consider creating
 * C-compatible FFI bindings or a REST API wrapper.
 *
 * Usage:
 *     php transcribe.php <audio_file>
 *     php transcribe.php --config /path/to/config.toml <audio_file>
 */

/**
 * Load configuration from a TOML file.
 */
function load_config(string $path): array {
    if (!file_exists($path)) {
        throw new RuntimeException("Config file not found: $path");
    }

    $content = file_get_contents($path);
    $config = [];
    $current_section = null;

    foreach (explode("\n", $content) as $line) {
        $line = trim($line);

        // Skip comments and empty lines
        if (empty($line) || $line[0] === '#') {
            continue;
        }

        // Section header
        if (preg_match('/^\[([^\]]+)\]$/', $line, $matches)) {
            $current_section = $matches[1];
            $config[$current_section] = [];
            continue;
        }

        // Key-value pair
        if (preg_match('/^([^=]+)=(.*)$/', $line, $matches)) {
            $key = trim($matches[1]);
            $value = trim($matches[2]);

            // Remove quotes
            $value = trim($value, '"\'');

            // Convert to appropriate type
            if ($value === 'true') $value = true;
            elseif ($value === 'false') $value = false;
            elseif (is_numeric($value)) $value = strpos($value, '.') !== false ? (float)$value : (int)$value;

            if ($current_section !== null) {
                $config[$current_section][$key] = $value;
            } else {
                $config[$key] = $value;
            }
        }
    }

    return $config;
}

/**
 * Transcription result structure.
 */
class TranscriptionResult {
    public string $content;
    public array $segments;
    public ?array $languages;
    public ?float $duration_seconds;
    public ?float $confidence;
    public ?int $speaker_count;

    public function __construct(array $data) {
        $this->content = $data['content'] ?? '';
        $this->segments = $data['segments'] ?? [];
        $this->languages = $data['languages'] ?? null;
        $this->duration_seconds = $data['duration_seconds'] ?? null;
        $this->confidence = $data['confidence'] ?? null;
        $this->speaker_count = $data['speaker_count'] ?? null;
    }
}

/**
 * Transcription client that wraps the Python bindings.
 */
class TranscriptionClient {
    private string $model_path;
    private string $python_script;

    public function __construct(string $model_path) {
        $this->model_path = $model_path;
        $this->python_script = __DIR__ . '/transcribe_helper.py';

        if (!file_exists($model_path)) {
            throw new RuntimeException("Model file not found: $model_path");
        }
    }

    public function transcribe(string $audio_path, ?string $language = null, int $speaker_count = 1): TranscriptionResult {
        if (!file_exists($audio_path)) {
            throw new RuntimeException("Audio file not found: $audio_path");
        }

        // Build command
        $cmd = sprintf(
            'python3 %s --model %s --audio %s --output json',
            escapeshellarg($this->python_script),
            escapeshellarg($this->model_path),
            escapeshellarg($audio_path)
        );

        if ($language) {
            $cmd .= ' --language ' . escapeshellarg($language);
        }

        if ($speaker_count > 1) {
            $cmd .= ' --speakers ' . escapeshellarg((string)$speaker_count);
        }

        // Execute
        $output = [];
        $return_code = 0;
        exec($cmd . ' 2>&1', $output, $return_code);

        if ($return_code !== 0) {
            throw new RuntimeException("Transcription failed: " . implode("\n", $output));
        }

        // Parse JSON output
        $json = implode("\n", $output);
        $data = json_decode($json, true);

        if ($data === null) {
            throw new RuntimeException("Failed to parse transcription output: $json");
        }

        return new TranscriptionResult($data);
    }
}

// Main
function main(array $argv): int {
    $args = array_slice($argv, 1);
    $config_path = __DIR__ . '/config.toml';

    // Parse --config option
    $config_idx = array_search('--config', $args);
    if ($config_idx !== false) {
        $config_path = $args[$config_idx + 1] ?? $config_path;
        array_splice($args, $config_idx, 2);
    }

    if (empty($args)) {
        echo "Usage: php transcribe.php [--config config.toml] <audio_file>\n";
        return 1;
    }

    $audio_file = $args[0];
    if (!file_exists($audio_file)) {
        echo "Error: Audio file not found: $audio_file\n";
        return 1;
    }

    // Load configuration
    try {
        $config = load_config($config_path);
    } catch (RuntimeException $e) {
        echo "Error: " . $e->getMessage() . "\n";
        return 1;
    }

    $model_path = $config['whisper']['model_path'] ?? '';
    $language = $config['transcription']['language'] ?? '';
    $speaker_count = $config['transcription']['speaker_count'] ?? 1;

    if (empty($model_path) || !file_exists($model_path)) {
        echo "Error: Whisper model not found: $model_path\n";
        echo "Download a model from: https://huggingface.co/ggerganov/whisper.cpp/tree/main\n";
        return 1;
    }

    // Create client and transcribe
    try {
        echo "Loading model: $model_path\n";
        $client = new TranscriptionClient($model_path);

        echo "\nTranscribing: $audio_file\n";
        echo str_repeat("-", 40) . "\n";

        $result = $client->transcribe(
            $audio_file,
            $language ?: null,
            $speaker_count
        );

        // Output results
        echo "\nContent:\n{$result->content}\n";

        if ($result->languages) {
            echo "\nDetected languages: " . implode(', ', $result->languages) . "\n";
        }

        if ($result->duration_seconds) {
            echo sprintf("Duration: %.2fs\n", $result->duration_seconds);
        }

        if (!empty($result->segments)) {
            echo sprintf("\nSegments (%d):\n", count($result->segments));
            foreach ($result->segments as $i => $seg) {
                $speaker = isset($seg['speaker']) ? " [{$seg['speaker']}]" : "";
                echo sprintf(
                    "  %d. [%.2fs - %.2fs]%s %s\n",
                    $i + 1,
                    $seg['start_seconds'],
                    $seg['end_seconds'],
                    $speaker,
                    $seg['text']
                );
            }
        }

        return 0;
    } catch (RuntimeException $e) {
        echo "Error: " . $e->getMessage() . "\n";
        return 1;
    }
}

exit(main($argv));
