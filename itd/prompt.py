class Prompter():
    
    def __init__(self, fore_prompt, post_prompt):
        self.fore_prompt = fore_prompt
        self.post_prompt = post_prompt
    
    def IO_prompt(self, xs):
        
        xs = [self.fore_prompt + "\n".join(xs) + self.post_prompt]
    
        return xs
    
    def GD_prompt(self, xs):
        
        xs = [self.fore_prompt + x + self.post_prompt for x in xs]
        
        return xs