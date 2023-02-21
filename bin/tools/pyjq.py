#!/usr/bin/env python3
import json
import subprocess
import pprint
import sys

output = json.loads(subprocess.check_output(executable='jq', args=sys.argv[1:], shell=True))
pprint.pprint(output)
