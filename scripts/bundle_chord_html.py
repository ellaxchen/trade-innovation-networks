import json, os

with open("../html/chord_data.json", encoding="utf-8") as f:
    data = json.load(f)

with open("../html/chord_interactive.html", encoding="utf-8") as f:
    html = f.read()

target = 'd3.json("chord_data.json").then(data => {'
if target not in html:
    print("ERROR: target string not found in HTML")
    exit(1)

# Replace opening
json_str = json.dumps(data)
html = html.replace(target, "const data = " + json_str + "; { // inline data")

# Replace the .then closing: });\n</script> -> }\n</script>
html = html.replace("});\n</script>", "}\n</script>")

out = "../html/chord_interactive_standalone.html"
with open(out, "w", encoding="utf-8") as f:
    f.write(html)

sz = os.path.getsize(out) / 1024
print(f"OK -> {out} ({sz:.0f} KB)")
