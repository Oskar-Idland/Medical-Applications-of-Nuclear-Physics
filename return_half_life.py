import curie as ci

def return_half_life(element: str | None = None, isotope: str | None = None) -> None:
    if element is None:
        element = input("Enter the element symbol (e.g., 'Ge', 'Ga'): ").strip().upper()
    if isotope is None:
        isotope = input("Enter the isotope mass number (e.g., '69', '71'): ").strip()
    print(element, isotope)
    iso_name = f"{element}-{isotope}"
    iso = ci.Isotope(iso_name)
    half_life = iso.half_life()
    if half_life == float('inf'):
        print(f"{iso_name} is stable and does not have any half-life.")
    else:
        print(half_life)


if __name__ == "__main__":
    return_half_life()