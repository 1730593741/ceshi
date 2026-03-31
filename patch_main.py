import pathlib
import sys

content = pathlib.Path('main.py').read_text(encoding='utf-8')
lines = content.splitlines(True)

new_lines = []
in_run_exp = False
try_added = False

for line in lines:
    if line.startswith("def run_experiment("):
        in_run_exp = True
        new_lines.append(line)
        continue
    
    if in_run_exp:
        if line.startswith("    _reset_run_logs("):
            new_lines.append("    try:\n")
            try_added = True
            
        if try_added:
            if not line.strip():
                new_lines.append(line)
            else:
                new_lines.append("    " + line)
                if line == "    return summary\n":
                    new_lines.append("    finally:\n")
                    new_lines.append("        runtime.runner.logger.close()\n")
                    new_lines.append("        if runtime.runner.experience_logger:\n")
                    new_lines.append("            runtime.runner.experience_logger.close()\n")
                    in_run_exp = False
                    try_added = False
        else:
            new_lines.append(line)
    else:
        new_lines.append(line)

pathlib.Path('main.py').write_text("".join(new_lines), encoding='utf-8')
print("Patched main.py successfully")
