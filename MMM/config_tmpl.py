from fitting.platform_manager import PlatformManager


class Config:
    db_connection_string = "DRIVER=" + PlatformManager.instance().driver + ";" \
                                                                           "Server=marketingsvr.database.windows.net;" \
                                                                           "UID=<username>;PWD=<password>;" \
                                                                           "Database=marketingdb;"
