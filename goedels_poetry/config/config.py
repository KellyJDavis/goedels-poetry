from configparser import ConfigParser

# Global ConfigParser
parsed_config = ConfigParser()

# Read config from '../data/config.ini' file
parsed_config.read("../data/config.ini")
