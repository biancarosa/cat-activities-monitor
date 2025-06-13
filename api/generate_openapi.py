#!/usr/bin/env python3
"""
OpenAPI Documentation Generator for Cat Activities Monitor

This script generates comprehensive OpenAPI documentation for the Cat Activities Monitor API
and saves it to both JSON and YAML formats for easy integration with documentation tools.
"""

import json
import yaml
from pathlib import Path
from main import app


def generate_openapi_docs():
    """Generate OpenAPI documentation and save to files."""
    
    # Get the OpenAPI schema from FastAPI
    openapi_schema = app.openapi()
    
    # Create docs directory if it doesn't exist
    docs_dir = Path("docs")
    docs_dir.mkdir(exist_ok=True)
    
    # Save as JSON
    json_path = docs_dir / "openapi.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(openapi_schema, f, indent=2, ensure_ascii=False)
    
    # Save as YAML
    yaml_path = docs_dir / "openapi.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(openapi_schema, f, default_flow_style=False, allow_unicode=True)
    
    # Generate a simple HTML documentation page
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Cat Activities Monitor API Documentation</title>
    <link rel="stylesheet" type="text/css" href="https://unpkg.com/swagger-ui-dist@5.9.0/swagger-ui.css" />
    <style>
        html {{
            box-sizing: border-box;
            overflow: -moz-scrollbars-vertical;
            overflow-y: scroll;
        }}
        *, *:before, *:after {{
            box-sizing: inherit;
        }}
        body {{
            margin:0;
            background: #fafafa;
        }}
    </style>
</head>
<body>
    <div id="swagger-ui"></div>
    <script src="https://unpkg.com/swagger-ui-dist@5.9.0/swagger-ui-bundle.js"></script>
    <script src="https://unpkg.com/swagger-ui-dist@5.9.0/swagger-ui-standalone-preset.js"></script>
    <script>
        window.onload = function() {{
            const ui = SwaggerUIBundle({{
                url: './openapi.json',
                dom_id: '#swagger-ui',
                deepLinking: true,
                presets: [
                    SwaggerUIBundle.presets.apis,
                    SwaggerUIStandalonePreset
                ],
                plugins: [
                    SwaggerUIBundle.plugins.DownloadUrl
                ],
                layout: "StandaloneLayout",
                tryItOutEnabled: true,
                supportedSubmitMethods: ['get', 'post', 'put', 'delete', 'patch'],
                docExpansion: 'list',
                defaultModelsExpandDepth: 2,
                defaultModelExpandDepth: 2
            }});
        }};
    </script>
</body>
</html>
    """
    
    html_path = docs_dir / "index.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    print(f"✅ OpenAPI documentation generated successfully!")
    print(f"📁 Files created in {docs_dir.absolute()}:")
    print(f"   - openapi.json ({json_path.stat().st_size} bytes)")
    print(f"   - openapi.yaml ({yaml_path.stat().st_size} bytes)")
    print(f"   - index.html (Swagger UI)")
    print(f"\n🌐 To view the documentation:")
    print(f"   - Open {html_path.absolute()} in your browser")
    print(f"   - Or start the API server and visit http://localhost:8000/docs")


if __name__ == "__main__":
    generate_openapi_docs() 