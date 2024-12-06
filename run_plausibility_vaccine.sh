#!/bin/bash

set -euo pipefail

print_usage() {
    echo "Usage: ${0} [-h] [config_name]"
    echo
    echo "Run Plausibility Vaccine experiments"
    echo
    echo "Positional argument:"
    echo "  config_file            The path of the configuration file to use (default='config/base.yaml')."
    echo
    echo "Optional arguments:"
    echo "  -h, --help             Show this help message and exit."
}

CONFIG_FILE="config/base.yaml"


main() {
    parse_args "${@}"
    echo "Running Plausibility Vaccine with config: ${CONFIG_FILE}..."
    uv run plausibility_vaccine/main.py --config-file ${CONFIG_FILE}
}

parse_args() {
    local i=0

    while [[ "${#}" -gt 0 ]]; do
        case "${1}" in
            -h|--help|help)
                print_usage
                exit 0
                ;;
            *)
                case "${i}" in
                    0)
                        CONFIG_FILE="${1}"
                        i=1
                        shift
                        ;;
                esac
                ;;
        esac
    done

    [[ -z "${CONFIG_FILE}" ]] && printf "Requires CONFIG_FILE\n\n" >&2 && print_usage >&2 && exit 1
    return 0
}


main "${@}"
