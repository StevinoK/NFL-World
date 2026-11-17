"""Launch the saved-estimate explorer on an available localhost port."""

from pathlib import Path
import os
import socket
import subprocess
import sys


def available_port(start=8510, attempts=100):
    for port in range(start, start + attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            if hasattr(socket, 'SO_EXCLUSIVEADDRUSE'):
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            try:
                probe.bind(('127.0.0.1', port))
                return port
            except OSError:
                continue
    raise RuntimeError(f'No available localhost port in {start}-{start + attempts - 1}.')


if __name__ == '__main__':
    app = Path(__file__).with_name('explorer.py')
    repo = app.resolve().parents[1]
    environment = {**os.environ, 'PYTHONPATH': os.pathsep.join(filter(None, [str(repo), os.environ.get('PYTHONPATH')]))}
    port = available_port()
    print(f'Player Estimates: http://127.0.0.1:{port}', flush=True)
    sys.exit(subprocess.run([
        sys.executable, '-m', 'streamlit', 'run', str(app),
        '--server.address', '127.0.0.1', '--server.port', str(port),
        '--server.headless', 'true', '--browser.gatherUsageStats', 'false',
        '--theme.base', 'light', '--theme.primaryColor', '#087f8c',
        '--theme.backgroundColor', '#ffffff', '--theme.textColor', '#263b3e',
    ], env=environment, cwd=str(repo), check=False).returncode)