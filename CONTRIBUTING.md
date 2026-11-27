
# Contributing to CloudBurst Intelligence System 🌩️

Thank you for your interest in contributing to CloudBurst!

We welcome developers, students, researchers, and open-source
enthusiasts to help improve our AI-powered cloudburst detection
and early warning system.

## 1. Getting Started

Before contributing:

- Read the project README.
- Explore the existing project structure.
- Check available GitHub issues.
- Comment on an issue before starting work.
- Wait for maintainer approval or assignment.

## 2. Development Setup

### Prerequisites

- Python 3.10 or a compatible project-supported version
- Git
- GitHub account

### Fork and Clone

Fork this repository on GitHub, then run:

```bash
git clone https://github.com/Nandaninaulaa/Cloudburst-intelligence-System.git
cd Cloudburst-intelligence-System
```

### Create a Virtual Environment

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Configure Environment Variables

Copy `.env.example` to `.env` and configure the required
API credentials.

Never commit your `.env` file or expose API keys.

### Run the Application

```bash
streamlit run src/prediction/app.py
```

## 3. Working on an Issue

1. Find an open issue.
2. Read its problem statement and acceptance criteria.
3. Comment explaining your proposed approach.
4. Wait for maintainer approval or assignment.
5. Work only on the agreed scope.

Avoid working on issues already assigned to someone else.

## 4. Branch Naming

Create a separate branch for each contribution.

Examples:

- feat/issue-12-weather-improvements
- fix/issue-15-api-security
- docs/issue-18-documentation

## 5. Coding Guidelines

- Follow Python PEP 8 conventions.
- Use descriptive variable and function names.
- Write modular and reusable code.
- Add comments where necessary.
- Handle errors appropriately.
- Avoid unnecessary dependencies.
- Never commit secrets, credentials, or personal data.

## 6. Testing

Before submitting a pull request:

- Run relevant tests.
- Check for errors and regressions.
- Verify that existing functionality still works.
- Add tests for new functionality wherever appropriate.

Run the test suite using:

```bash
python -m pytest
```

## 7. Pull Request Guidelines

Your pull request should include:

- A clear and descriptive title.
- A summary of changes.
- The related issue number.
- Testing performed.
- Screenshots for UI changes, where relevant.

Link the issue using:

Closes #ISSUE_NUMBER

Keep pull requests focused and avoid unrelated changes.

## 8. Review Process

Maintainers will review submitted pull requests.

You may be asked to:

- Fix bugs.
- Improve code quality.
- Add missing tests.
- Update documentation.
- Make changes based on review feedback.

Please respond constructively to review comments.

## 9. Winter of Code Social

For WOCS contributions:

- Contributors must register for WOCS.
- PRs must satisfy the project's contribution requirements.
- Project Admins or Mentors assign the final difficulty level.
- Eligible merged PRs require the correct WOCS labels.

The Project Admin or Mentor is responsible for assigning
the WOCS labels before merging.

## 10. Code of Conduct

Be respectful, collaborative, and constructive.

Help beginners, communicate clearly, and maintain a
welcoming open-source environment.

Thank you for helping build CloudBurst! 💙
