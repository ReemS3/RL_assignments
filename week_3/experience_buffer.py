class ExperienceBuffer:
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = []

    def push(self, element):
        """pushes the element inside of the buffer in case the size is not
        exhauseted, otherwise the oldest element is deleted"""
        if len(self.buffer) == self.max_size:
            self.buffer.pop(0)
        self.buffer.append(element)

    def pop(self):
        """returns and deletes the first/oldest element"""
        return self.buffer.pop(0)

    def get_length(self):
        """get the length of the buffere"""
        return len(self.buffer)
