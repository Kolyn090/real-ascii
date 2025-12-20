import os.path

from writer import PositionalCharTemplate, CharTemplate

class AsciiWriter:
    def __init__(self, p_cts: list[PositionalCharTemplate],
                 width: int, save_path=''):
        self.p_cts = p_cts
        self.width = width
        self.save_path = save_path
        p_cts.sort(key=lambda x: (x.top_left[1], x.top_left[0]))
        self.output_folder = 'f_output'
        self.chars_file = 'chars.txt'

    def save(self):
        if not os.path.exists(self.save_path):
            print("ASCII Writer: Invalid save path, abort.")
        output_path = self.get_output_path()
        os.makedirs(output_path, exist_ok=True)
        chars_path = os.path.join(output_path, self.chars_file)

        content = self._get_2d()
        with open(chars_path, "w", encoding="utf-8") as f:
            for row in content:
                f.write("".join(row) + "\n")

    def get_output_path(self):
        return os.path.join(self.save_path, self.output_folder)

    def _get_2d(self) -> list[list[str]]:
        chars = [p_ct.char_template.char for p_ct in self.p_cts]
        return [chars[i:i + self.width] for i in range(0, len(chars), self.width)]

    def _save_chars(self):
        pass

    def _save_color(self):
        # No support
        pass

def test():
    p_ct1 = PositionalCharTemplate(
        CharTemplate('a', None, None, None, None),
        (13, 22)
    )
    p_ct2 = PositionalCharTemplate(
        CharTemplate('b', None, None, None, None),
        (0, 22)
    )
    p_ct3 = PositionalCharTemplate(
        CharTemplate('c', None, None, None, None),
        (13, 0)
    )
    p_ct4 = PositionalCharTemplate(
        CharTemplate('d', None, None, None, None),
        (0, 0)
    )
    p_cts = [p_ct1, p_ct2, p_ct3, p_ct4]
    ascii_writer = AsciiWriter(p_cts, 1)
    for p_ct in ascii_writer.p_cts:
        print(p_ct)
    print(ascii_writer._get_2d())

    ascii_writer = AsciiWriter(p_cts, 2, './')
    print(ascii_writer._get_2d())
    ascii_writer.save()

if __name__ == '__main__':
    test()
