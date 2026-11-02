import React from 'react';

export interface ButtonProps {
  label: string;
  onClick: () => void;
}

// Базовый UI-компонент кнопки
export const Button: React.FC<ButtonProps> = ({ label, onClick }) => {
  return (
    <button className="primary-btn" onClick={onClick}>
      {label}
    </button>
  );
};
