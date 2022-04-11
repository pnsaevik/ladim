class Legacy_IBM:
    def __init__(self, modules):
        self.modules = modules
        config = modules['config']

        if config["ibm_module"]:
            # Import the module
            import logging
            import sys
            import os
            import importlib
            logging.info("Initializing the IBM")
            sys.path.insert(0, os.getcwd())
            ibm_module = importlib.import_module(config["ibm_module"])
            # Initiate the IBM object
            self.ibm = ibm_module.IBM(config)

        else:
            self.ibm = None

    def update(self):
        if self.ibm is not None:
            self.ibm.update_ibm(
                self.modules['grid'],
                self.modules['state'],
                self.modules['forcing'],
            )
