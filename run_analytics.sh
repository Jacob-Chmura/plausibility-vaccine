#!/bin/bash

set -euo pipefail

print_usage() {
    echo "Usage: ${0} [-h]"
    echo
    echo "Run Plausibility Vaccine analytics"
    echo
    echo "Positional argument:"
    echo
    echo "Optional arguments:"
    echo "  --no-results           Skip analytics that require having previously run Plausibility Vaccine."
    echo "  -h, --help             Show this help message and exit."
}


RESULTS=1

main() {
    parse_args "${@}"
    echo "Running Plausibility Vaccine Analytics..."

    if [[ ${RESULTS} -eq 0 ]]; then
        echo "Skipping analytics that require having previously run Plausibility Vaccine!"
    fi

    echo "Running data analysis"
    uv run scripts/run_data_analysis.py

    echo "Running word sense analysis"
    uv run scripts/run_word_sense_analysis.py

    echo "Running property-plausibility mutual information analysis"
    uv run scripts/run_property_plausibility_mutual_info.py

    if [[ ${RESULTS} -eq 0 ]]; then
        exit 0
    fi

    echo "Running performance analysis"
    uv run scripts/aggregate_performance_results.py
}

parse_args() {
    local i=0

    while [[ "${#}" -gt 0 ]]; do
        case "${1}" in
            -h|--help|help)
                print_usage
                exit 0
                ;;
            --no-results)
                RESULTS=0
                shift
                ;;
            *)
                shift
        esac
    done

    return 0
}


main "${@}"