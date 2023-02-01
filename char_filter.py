import glob, json

def char_filter(data_path: str) -> tuple(str, list(str)):
    """
    Loads all text files from data_path and returns a list of all unique characters
    Also returns a list of texts gathered from the input files
    """
    files = glob.glob(data_path + '/*.txt')
    files_text = []
    text = ""
    for f in files:
        with open(f, 'r', encoding='utf-8') as fp:
            file = fp.read()
            text += file
            files_text.append(file)
            
    # here are all the unique characters that occur in this text
    chars = sorted(list(set(text)))
    chars_with_count = { ch:text.count(ch) for ch in chars }
    # sort by count
    chars_with_count = {k: v for k, v in sorted(chars_with_count.items(), key=lambda item: item[1], reverse=True)}

    charlist = ""
    skipped = ""
    skipped_count = 0
    for c in chars_with_count.keys():
        if chars_with_count[c] > 100:
            charlist += c
        else:
            skipped += c
            skipped_count += chars_with_count[c]

    print("old:", len(chars_with_count), "-- new:", len(charlist))
    print("with occurrence > 100:", json.dumps(charlist, ensure_ascii=False))
    checks = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,;:!?\"'()[]{}-–—_#&@£$€¥¢^°+=*~|/\\<>%©®™"

    too_few_occurrences = ""
    for c in checks:
        if c not in charlist:
            too_few_occurrences += c

    print(f'not included: "{"".join(too_few_occurrences)}"', )
    merged_printable = json.dumps(charlist, ensure_ascii=False)[1:-1] + json.dumps(too_few_occurrences, ensure_ascii=False)[1:-1]
    print(f'merged: "{merged_printable}"')

    merged = charlist + too_few_occurrences

    included_chars = 0
    for c in merged:
        if c in chars_with_count:
            included_chars += chars_with_count[c]

    total = 0
    for c in chars_with_count:
        total += chars_with_count[c]

    diff = total - included_chars
    print(f"total: {total}, included: {included_chars}, diff: {diff} {skipped_count}") # todo test skipped_count
    
    return merged, files_text