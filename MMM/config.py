from fitting.platform_manager import PlatformManager


class Config:

    db_connection_string = "DRIVER=" + PlatformManager.instance().driver + ";" \
                           "Server=marketingsvr.database.windows.net;" \
                           "UID=vinay.suryaprakash;PWD=2yseKU4S7xi8F*skjRCrSZjd;" \
                           "Database=marketingdb;"