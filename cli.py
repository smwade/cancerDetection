import click
from mediaug.dataset import Dataset

@click.command()
def random():
    pass

@click.group()
def train():
    pass

@train.command()
def pix2pix():
    """Command on cli1"""

@train.command()
def singan():
    """Command on cli1"""

@click.group()
def test():
    pass

@test.command()
def pix2pix_test():
    """Cond on cli1"""

@test.command()
def singan_test():
    """Command1"""

cli = click.CommandCollection(sources=[train, test])

if __name__ == '__main__':
    cli()
