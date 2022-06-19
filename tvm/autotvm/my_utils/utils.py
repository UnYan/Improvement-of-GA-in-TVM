
# 根据字典转换为对应配置类
class Utils:
    @staticmethod
    def parseDict(o:object, dic:dict, map):
        if o is None or dic is None or type(dic) is not dict:
            return

        for key, value in dic.items():
            if hasattr(o, "_" + o.__class__.__name__ +"__" + key) and key in map.keys():
                setMethod = getattr(o, map[key])
                setMethod(value)