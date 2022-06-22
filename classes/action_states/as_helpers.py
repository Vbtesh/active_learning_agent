
# Peusdo action class to manage local experiments in tree search active states
# Can be instantiated with attributes needed in internal states updates and passed as argument in the update method
class Pseudo_AS():
    def __init__(self, action, action_length, realised):
        self.a = action        
        self.a_len = action_length

        if realised:
            self.realised = True
            self.a_real = action
        else:
            self.realised = False