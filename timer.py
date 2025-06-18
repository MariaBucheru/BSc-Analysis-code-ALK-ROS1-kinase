import time


class Timer(list):
    def __call__(self, msg=None, nopar=False):
                    
        # With a message, run as context
        if msg is not None:
            self.append([msg, time.time()])
            return self
        
        # Without a message, run as decorator
        def inner(func):
            run = [0]
            def wrapper(*args, **kwargs):
                run.append(run[-1] + 1)
                msg = f'  {func.__name__}[{run[-1]}]'
                msg += ' ...' if nopar else f'{args} {kwargs}'
                with self(msg=msg):
                    return func(*args, **kwargs)
            return wrapper
        return inner
        
    def __repr__(self):
        if not hasattr(self, 'idx'):
            self.idx = 0    
        out = '\n'.join(f'{lbl}: {e-s:.6f}s' for lbl, s, e in self[self.idx:])
        self.idx = len(self)
        
        return out
    
    def __enter__(self):
        pass
    
    def __exit__(self, exc_type, exc_value, exc_tb):
        for item in self[::-1]:
            if len(item) == 2:
                item.append(time.time())
                break
        print(self)
