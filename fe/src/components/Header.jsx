import Logo from "./Logo.jsx";

export default function Header({ room, time }) {
  return (
    <header className="header">
      <div className="brand" aria-label={`SeeCure ${room.name}`}>
        <Logo />
      </div>
      <time className="header-time">{time}</time>
    </header>
  );
}
