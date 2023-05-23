

make_repo:
	@-export NAME=$$(basename -s .git `git config --get remote.origin.url`); while true; do find . -exec bash -c 'git mv $$0 $$(echo $$0 | sed -E "s;___package_name___;$$NAME;g") 2> /dev/null' {} \; 2> /dev/null && break; done; 
	@export NAME=$$(basename -s .git `git config --get remote.origin.url`); find . -type f ! -name "Makefile" ! -path "./.*" -exec bash -c 'sed -i -E "s;___package_name___;$$NAME;g" $$0' {} \;  
	@git commit -a -m "Initial template renaming" 
	pip install -e .
