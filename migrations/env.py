from alembic import context
from cagemetrics.config import Settings
from cagemetrics.db import Base, make_engine
import cagemetrics.forward  # noqa: F401 -- register prospective ledger metadata

config = context.config
config.set_main_option("sqlalchemy.url", Settings.from_env().database_url.replace("%", "%%"))
target_metadata = Base.metadata

if context.is_offline_mode():
    context.configure(
        url=Settings.from_env().database_url, target_metadata=target_metadata, literal_binds=True
    )
    with context.begin_transaction():
        context.run_migrations()
else:
    engine = make_engine(Settings.from_env().database_url)
    with engine.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            render_as_batch=engine.dialect.name == "sqlite",
        )
        with context.begin_transaction():
            context.run_migrations()
    engine.dispose()
