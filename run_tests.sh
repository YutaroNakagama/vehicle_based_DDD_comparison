#!/usr/bin/env bash
# Quick test runner script for development
# Usage: ./run_tests.sh [options]

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Running DDD Test Suite${NC}"

# Default: run fast tests only
if [ $# -eq 0 ]; then
    echo -e "${YELLOW}Running fast tests (unit tests, no slow tests)${NC}"
    pytest -m "unit and not slow" -n auto --tb=short
else
    case "$1" in
        --all)
            echo -e "${YELLOW}Running all tests${NC}"
            pytest -n auto
            ;;
        --fast)
            echo -e "${YELLOW}Running only fast tests${NC}"
            pytest -m "not slow" -n auto --tb=short
            ;;
        --unit)
            echo -e "${YELLOW}Running unit tests only${NC}"
            pytest -m "unit" -n auto --tb=short
            ;;
        --integration)
            echo -e "${YELLOW}Running integration tests${NC}"
            pytest -m "integration" --tb=short
            ;;
        --coverage)
            echo -e "${YELLOW}Running tests with coverage${NC}"
            pytest --cov=src --cov-report=term-missing --cov-report=html
            echo -e "${GREEN}Coverage report saved to htmlcov/index.html${NC}"
            ;;
        --help)
            echo "Usage: ./run_tests.sh [option]"
            echo ""
            echo "Options:"
            echo "  (no args)       Run fast unit tests (default)"
            echo "  --all          Run all tests including slow ones"
            echo "  --fast         Run all tests except slow ones"
            echo "  --unit         Run only unit tests"
            echo "  --integration  Run only integration tests"
            echo "  --coverage     Run tests with coverage report"
            echo "  --help         Show this help message"
            ;;
        *)
            echo -e "${YELLOW}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
fi

echo -e "${GREEN}Tests completed!${NC}"
