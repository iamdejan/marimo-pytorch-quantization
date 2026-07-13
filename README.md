---
title: Readme
marimo-version: 0.23.13
---

# Marimo Pixi Template

A template repository for Marimo project, managed by Pixi.

## Prerequisites

Before running this program, ensure you have the following installed:

1. **Pixi** - A package manager for Rust and other languages. Install it by following the official guide at [https://pixi.prefix.dev/latest/](https://pixi.prefix.dev/latest/).

2. **Download Dependencies** - Once Pixi is installed, run the following command to download all project dependencies:
   ```bash
   pixi install
   ```

## Run the Program

Use Pixi to run it:
   ```bash
   pixi run start
   ```

### Available Commands

| Command | Description |
|---------|-------------|
| `pixi run lint` | Run lint check using marimo check |
| `pixi run lint-fix` | Auto-fix linting issues using marimo check |
