class Legacy_IBM:
    def __init__(self, modules, **config):
        self.modules = modules

        legacy_conf = dict(
            ibm=config,
            dt=config['dt'],
            start_time=config['start_time'],
            output_instance=config['output_instance'],
            nc_attributes=config['nc_attributes'],
        )

        if config["legacy_module"] is not None:
            # Import the module
            import logging
            import sys
            import os
            import importlib
            logging.info("Initializing the IBM")
            sys.path.insert(0, os.getcwd())
            ibm_module = importlib.import_module(config["legacy_module"])
            # Initiate the IBM object
            self.ibm = ibm_module.IBM(legacy_conf)

        else:
            self.ibm = None

    def update(self):
        if self.ibm is not None:
            self.ibm.update_ibm(
                self.modules['grid'],
                self.modules['state'],
                self.modules['forcing'],
            )
