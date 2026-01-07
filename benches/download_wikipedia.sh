#!/bin/bash
# Download Wikipedia Cirrus dumps for benchmarking
# Uses pre-processed dumps with templates already expanded

set -e

DATA_DIR="${1:-$(dirname "$0")/data}"
mkdir -p "$DATA_DIR"

echo "=== Wikipedia Download Script ==="
echo "Data directory: $DATA_DIR"
echo

# Function to download and extract a dump
download_wiki() {
    local wiki="$1"
    local desc="$2"
    
    # Get the latest dump date
    local base_url="https://dumps.wikimedia.org/other/cirrussearch"
    local latest=$(curl -s "$base_url/" | grep -oP '\d{8}' | sort -r | head -1)
    
    if [ -z "$latest" ]; then
        echo "ERROR: Could not find latest dump date"
        return 1
    fi
    
    local dump_url="${base_url}/${latest}/${wiki}-${latest}-cirrussearch-content.json.gz"
    local output_gz="$DATA_DIR/${wiki}-content.json.gz"
    local output_txt="$DATA_DIR/${wiki}-content.txt"
    
    echo "=== $desc ==="
    echo "URL: $dump_url"
    
    if [ -f "$output_txt" ]; then
        local size=$(du -h "$output_txt" | cut -f1)
        echo "Already exists: $output_txt ($size)"
        return 0
    fi
    
    echo "Downloading..."
    if ! curl -L --progress-bar -o "$output_gz" "$dump_url"; then
        echo "ERROR: Download failed"
        return 1
    fi
    
    local gz_size=$(du -h "$output_gz" | cut -f1)
    echo "Downloaded: $gz_size (compressed)"
    
    echo "Extracting and converting to plain text..."
    # Extract just the text field from each JSON line
    gunzip -c "$output_gz" | \
        jq -r 'select(.text != null) | .text' 2>/dev/null | \
        head -c $((10 * 1024 * 1024 * 1024)) > "$output_txt" || true
    
    local txt_size=$(du -h "$output_txt" | cut -f1)
    echo "Extracted: $txt_size"
    
    # Keep gz for re-extraction if needed
    echo "Done: $output_txt"
    echo
}

# Check for jq
if ! command -v jq &> /dev/null; then
    echo "ERROR: jq is required for JSON parsing"
    echo "Install with: sudo apt install jq  OR  brew install jq"
    exit 1
fi

# Parse arguments
case "${2:-simple}" in
    simple)
        download_wiki "simplewiki" "Simple English Wikipedia (~1GB)"
        ;;
    en-sample)
        # Download English but limit to first 10GB
        download_wiki "enwiki" "English Wikipedia (sample, ~10GB)"
        ;;
    en)
        download_wiki "enwiki" "Full English Wikipedia (~80GB)"
        ;;
    all)
        download_wiki "simplewiki" "Simple English Wikipedia (~1GB)"
        download_wiki "enwiki" "Full English Wikipedia (~80GB)"
        ;;
    *)
        echo "Usage: $0 [data_dir] [simple|en-sample|en|all]"
        echo
        echo "Options:"
        echo "  simple    - Simple English Wikipedia (~636MB gz, ~1GB txt)"
        echo "  en-sample - English Wikipedia first 10GB"
        echo "  en        - Full English Wikipedia (~20GB gz, ~80GB txt)"  
        echo "  all       - Download all of the above"
        exit 1
        ;;
esac

echo "=== Download complete ==="
echo "Files in $DATA_DIR:"
ls -lh "$DATA_DIR"/*.txt 2>/dev/null || echo "(none)"
