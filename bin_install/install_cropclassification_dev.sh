#!/bin/bash

scriptdir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
"$scriptdir/install_cropclassification.sh" --envname cropclassificationdev --fordev Y
