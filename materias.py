import pdfplumber

def clean_text(text):
    cleaned_text = text.replace('\n', ' ').replace('|', '').strip()
    return cleaned_text

with pdfplumber.open('materias.pdf') as pdf:
    page = pdf.pages[1]
    table = page.extract_table()

    cleaned_table = []

    for row in table:
        cleaned_row = [clean_text(cell) for cell in row]
        cleaned_table.append(cleaned_row)

    # Guardar la tabla limpia en un archivo de texto
    with open('cleaned_table.txt', 'w') as file:
        for row in cleaned_table:
            # Convertir cada celda de la fila en una cadena y unirlas con un separador (p. ej. una coma)
            row_text = ', '.join(row)
            
            # Escribir la fila en el archivo de texto y agregar un salto de línea al final
            file.write(row_text + '\n')
