package main

import (
	"context"
	"fmt"
	"os"

	"dagger/opencode-dspy/internal/dagger"
)

type OpencodeDspy struct{}

func baseContainer(source *dagger.Directory) *dagger.Container {
	return dag.Container().
		From("ghcr.io/astral-sh/uv:python3.12-bookworm").
		WithMountedCache("/root/.cache/uv", dag.CacheVolume("uv-cache")).
		WithDirectory("/src", source).
		WithWorkdir("/src/dspy-trainingv2").
		WithExec([]string{"uv", "pip", "install", "--system", "-r", "requirements.txt"})
}

func (m *OpencodeDspy) Test(ctx context.Context, source *dagger.Directory) (string, error) {
	output, err := baseContainer(source).
		WithExec([]string{"python", "-m", "pytest", "tests/", "-v"}).
		Stdout(ctx)

	if err != nil {
		return "", fmt.Errorf("tests failed: %w", err)
	}

	return output, nil
}

func (m *OpencodeDspy) Validate(ctx context.Context, source *dagger.Directory) (string, error) {
	output, err := baseContainer(source).
		WithExec([]string{"python", "cli.py", "validate"}).
		Stdout(ctx)

	if err != nil {
		return "", fmt.Errorf("validation failed: %w", err)
	}

	return output, nil
}

func (m *OpencodeDspy) All(ctx context.Context, source *dagger.Directory) (string, error) {
	testOutput, err := m.Test(ctx, source)
	if err != nil {
		return "", err
	}

	validateOutput, err := m.Validate(ctx, source)
	if err != nil {
		return "", err
	}

	return fmt.Sprintf("Tests:\n%s\n\nValidation:\n%s", testOutput, validateOutput), nil
}

func (m *OpencodeDspy) Train(ctx context.Context, source *dagger.Directory, experimentName string) (string, error) {
	openaiKey := os.Getenv("OPENAI_API_KEY")
	if openaiKey == "" {
		return "", fmt.Errorf("OPENAI_API_KEY environment variable not set. Set it with: export OPENAI_API_KEY=sk-...")
	}

	openaiSecret := dag.SetSecret("OPENAI_API_KEY", openaiKey)

	output, err := baseContainer(source).
		WithSecretVariable("OPENAI_API_KEY", openaiSecret).
		WithExec([]string{"python", "cli.py", "train", "--experiment-name", experimentName}).
		Stdout(ctx)

	if err != nil {
		return "", fmt.Errorf("training failed: %w", err)
	}

	return output, nil
}
