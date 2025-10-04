## Running the Project with Docker

This project is containerized using Docker and Docker Compose for reproducible environments and simplified setup.

### Project-Specific Requirements
- **Python Version:** 3.13 (as specified in the Dockerfile)
- **Dependencies:** Installed in a virtual environment within the container. Key packages include:
  - yfinance
  - matplotlib
  - pandas
  - gdeltdoc
  - vaderSentiment
  - scikit-learn
  - seaborn
  - scipy
  - mariadb
  - sqlalchemy
- **System Libraries:** The container installs MariaDB client libraries and build tools for scientific packages.

### Environment Variables
- No environment variables are strictly required for the Python app by default.
- If you use a `.env` file for configuration, uncomment the `env_file` line in `docker-compose.yml` and provide your `.env` file in the project root.
- If you enable the MariaDB service, set the following environment variables in `docker-compose.yml`:
  - `MYSQL_ROOT_PASSWORD` (required, change to a secure value)
  - `MYSQL_DATABASE`, `MYSQL_USER`, `MYSQL_PASSWORD` (customize as needed)

### Build and Run Instructions
1. **Build and start the application:**
   ```sh
   docker compose up --build
   ```
   This will build the Python app image and start the container.

2. **(Optional) Enable MariaDB:**
   - Uncomment the `mariadb` service in `docker-compose.yml` if your `main.py` requires a database.
   - Uncomment the `depends_on` and `networks` sections for the Python app as needed.
   - Uncomment the `volumes` and `networks` sections at the bottom of the compose file for persistent storage and inter-service communication.

### Special Configuration
- The Python app runs as a non-root user (`appuser`) for improved security.
- All Python dependencies are installed in an isolated virtual environment inside the container.
- If your app exposes a web server, add the appropriate `ports` mapping in `docker-compose.yml` under the `python-app` service (e.g., `"8000:8000"`).

### Ports
- **No ports are exposed by default.**
- If your app provides a web interface or API, add a `ports` section to the `python-app` service in `docker-compose.yml`.
- If MariaDB is enabled, it uses the default internal port (3306), but is not exposed outside the Docker network unless you add a `ports` mapping.

---

**Note:** Only `main.py` is included in the container build. The `.git` directory and any secrets are excluded for security.